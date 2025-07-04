#include <luisa/xir/metadata/comment.h>
#include <luisa/core/logging.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/ast/external_function.h>
#include <luisa/ast/statement.h>
#include <luisa/ast/function.h>
#include <luisa/xir/builder.h>
#include <luisa/xir/special_register.h>
#include <luisa/xir/translators/ast2xir.h>

namespace luisa::compute::xir {

class AST2XIRContext {

public:
    struct BreakContinueTarget {
        BasicBlock *break_target{nullptr};
        BasicBlock *continue_target{nullptr};
    };

    struct Current {
        FunctionDefinition *f{nullptr};
        const ASTFunction *ast{nullptr};
        BreakContinueTarget break_continue_target;
        luisa::unordered_map<Variable, Value *> variables;
        luisa::vector<const CommentStmt *> comments;
    };

    struct TypedLiteral {
        const Type *type;
        LiteralExpr::Value value;

        [[nodiscard]] bool operator==(const TypedLiteral &rhs) const noexcept {
            return this->type == rhs.type &&
                   this->value.index() == rhs.value.index() &&
                   luisa::visit(
                       [&rhs_variant = rhs.value]<typename T>(T lhs) noexcept -> bool {
                           auto rhs = luisa::get<T>(rhs_variant);
                           if constexpr (luisa::is_scalar_v<T>) {
                               return lhs == rhs;
                           } else if constexpr (luisa::is_vector_v<T>) {
                               return luisa::all(lhs == rhs);
                           } else if constexpr (luisa::is_matrix_v<T>) {
                               for (auto i = 0u; i < luisa::matrix_dimension_v<T>; i++) {
                                   if (!all(lhs[i] == rhs[i])) { return false; }
                               }
                               return true;
                           }
                           LUISA_ERROR_WITH_LOCATION("Unexpected literal type.");
                       },
                       this->value);
        }

        [[nodiscard]] uint64_t hash() const noexcept {
            auto hv = luisa::visit(
                []<typename T>(T v) noexcept {
                    return hash_value(v);
                },
                value);
            return hash_combine({type->hash(), hv});
        }
    };

private:
    AST2XIRConfig _config;
    luisa::unique_ptr<Module> _module;
    luisa::unordered_map<uint64_t, Function *> _generated_functions;
    luisa::unordered_map<ConstantData, Constant *> _generated_constants;
    luisa::unordered_map<TypedLiteral, Constant *> _generated_literals;
    luisa::unordered_map<const Type *, Constant *> _generated_zero_constants;
    luisa::unordered_map<const Type *, Constant *> _generated_one_constants;
    Current _current;

private:
    [[nodiscard]] Value *_translate_unary_expr(XIRBuilder &b, const UnaryExpr *expr) noexcept {
        auto operand = _translate_expression(b, expr->operand(), true);
        // matrices need special handling
        if (operand->type()->is_matrix()) {
            switch (expr->op()) {
                case UnaryOp::PLUS: return operand;
                case UnaryOp::MINUS: return b.call(expr->type(), ArithmeticOp::MATRIX_COMP_NEG, {operand});
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Invalid unary operation.");
        }
        // normal cases
        auto op = [unary_op = expr->op()] {
            switch (unary_op) {
                case UnaryOp::PLUS: return ArithmeticOp::UNARY_PLUS;
                case UnaryOp::MINUS: return ArithmeticOp::UNARY_MINUS;
                case UnaryOp::NOT: return ArithmeticOp::UNARY_BIT_NOT;
                case UnaryOp::BIT_NOT: return ArithmeticOp::UNARY_BIT_NOT;
            }
            LUISA_ERROR_WITH_LOCATION("Unexpected unary operation.");
        }();
        if (expr->op() == UnaryOp::NOT) {
            LUISA_DEBUG_ASSERT(expr->type()->is_bool() || expr->type()->is_bool_vector(),
                               "Invalid type for logical not operation.");
            operand = _type_cast_if_necessary(b, expr->type(), operand);
        }
        return b.call(expr->type(), op, {operand});
    }

    [[nodiscard]] Value *_type_cast_if_necessary(XIRBuilder &b, const Type *type, Value *value) noexcept {
        // no cast needed
        if (type == value->type()) { return value; }
        // scalar to scalar cast
        if (value->type()->is_scalar() && type->is_scalar()) {
            return b.static_cast_(type, value);
        }
        // vector to vector cast
        if (value->type()->is_vector() && type->is_vector()) {
            LUISA_ASSERT(value->type()->dimension() >= type->dimension(), "Vector cast dimension mismatch.");
            auto value_elem_type = value->type()->element();
            luisa::fixed_vector<Value *, 4u> elements;
            for (auto i = 0u; i < type->dimension(); i++) {
                auto idx = _translate_constant_access_index(i);
                auto elem = b.call(value_elem_type, ArithmeticOp::EXTRACT, {value, idx});
                elements.emplace_back(b.static_cast_if_necessary(type->element(), elem));
            }
            return b.call(type, ArithmeticOp::AGGREGATE, elements);
        }
        // scalar to vector cast
        if (value->type()->is_scalar() && type->is_vector()) {
            value = b.static_cast_if_necessary(type->element(), value);
            luisa::fixed_vector<Value *, 4u> elements;
            for (auto i = 0u; i < type->dimension(); i++) { elements.emplace_back(value); }
            return b.call(type, ArithmeticOp::AGGREGATE, elements);
        }
        LUISA_ERROR_WITH_LOCATION("Invalid cast operation.");
    }

    [[nodiscard]] Value *_translate_binary_expr(XIRBuilder &b, const BinaryExpr *expr) noexcept {
        auto type_promotion = promote_types(expr->op(), expr->lhs()->type(), expr->rhs()->type());
        auto op = [binary_op = expr->op(), lhs = expr->lhs(), rhs = expr->rhs()] {
            auto has_matrix = lhs->type()->is_matrix() || rhs->type()->is_matrix();
            switch (binary_op) {
                case BinaryOp::ADD: return has_matrix ? ArithmeticOp::MATRIX_COMP_ADD : ArithmeticOp::BINARY_ADD;
                case BinaryOp::SUB: return has_matrix ? ArithmeticOp::MATRIX_COMP_SUB : ArithmeticOp::BINARY_SUB;
                case BinaryOp::MUL: {
                    if (lhs->type()->is_matrix() && (rhs->type()->is_matrix() || rhs->type()->is_vector())) {
                        return ArithmeticOp::MATRIX_LINALG_MUL;
                    }
                    return has_matrix ? ArithmeticOp::MATRIX_COMP_MUL : ArithmeticOp::BINARY_MUL;
                }
                case BinaryOp::DIV: return has_matrix ? ArithmeticOp::MATRIX_COMP_DIV : ArithmeticOp::BINARY_DIV;
                case BinaryOp::MOD: return ArithmeticOp::BINARY_MOD;
                case BinaryOp::BIT_AND: return ArithmeticOp::BINARY_BIT_AND;
                case BinaryOp::BIT_OR: return ArithmeticOp::BINARY_BIT_OR;
                case BinaryOp::BIT_XOR: return ArithmeticOp::BINARY_BIT_XOR;
                case BinaryOp::SHL: return ArithmeticOp::BINARY_SHIFT_LEFT;
                case BinaryOp::SHR: return ArithmeticOp::BINARY_SHIFT_RIGHT;
                case BinaryOp::AND: return ArithmeticOp::BINARY_BIT_AND;
                case BinaryOp::OR: return ArithmeticOp::BINARY_BIT_OR;
                case BinaryOp::LESS: return ArithmeticOp::BINARY_LESS;
                case BinaryOp::GREATER: return ArithmeticOp::BINARY_GREATER;
                case BinaryOp::LESS_EQUAL: return ArithmeticOp::BINARY_LESS_EQUAL;
                case BinaryOp::GREATER_EQUAL: return ArithmeticOp::BINARY_GREATER_EQUAL;
                case BinaryOp::EQUAL: return ArithmeticOp::BINARY_EQUAL;
                case BinaryOp::NOT_EQUAL: return ArithmeticOp::BINARY_NOT_EQUAL;
            }
            LUISA_ERROR_WITH_LOCATION("Unexpected binary operation.");
        }();
        auto lhs = _translate_expression(b, expr->lhs(), true);
        auto rhs = _translate_expression(b, expr->rhs(), true);
        if (expr->op() == BinaryOp::AND || expr->op() == BinaryOp::OR) {
            LUISA_DEBUG_ASSERT(type_promotion.result->is_bool() ||
                                   type_promotion.result->is_bool_vector(),
                               "Invalid type promotion result type for binary logic operator: {}.",
                               type_promotion.result->description());
            lhs = b.static_cast_if_necessary(type_promotion.result, lhs);
            rhs = b.static_cast_if_necessary(type_promotion.result, rhs);
        } else {
            lhs = _type_cast_if_necessary(b, type_promotion.lhs, lhs);
            rhs = _type_cast_if_necessary(b, type_promotion.rhs, rhs);
        }
        auto result = b.call(expr->type(), op, {lhs, rhs});
        return _type_cast_if_necessary(b, type_promotion.result, result);
    }

    [[nodiscard]] Value *_translate_constant_access_index(uint i) noexcept {
        auto key = TypedLiteral{Type::of<uint>(), LiteralExpr::Value{i}};
        return _translate_typed_literal(key);
    }

    [[nodiscard]] Value *_collect_access_indices(XIRBuilder &b, const Expression *expr, luisa::fixed_vector<Value *, 16u> &rev_indices) noexcept {
        switch (expr->tag()) {
            case Expression::Tag::MEMBER: {
                auto member_expr = static_cast<const MemberExpr *>(expr);
                auto member_index = _translate_constant_access_index(member_expr->member_index());
                rev_indices.emplace_back(member_index);
                return _collect_access_indices(b, member_expr->self(), rev_indices);
            }
            case Expression::Tag::ACCESS: {
                auto access_expr = static_cast<const AccessExpr *>(expr);
                auto index = _translate_expression(b, access_expr->index(), true);
                rev_indices.emplace_back(index);
                return _collect_access_indices(b, access_expr->range(), rev_indices);
            }
            default: break;
        }
        return _translate_expression(b, expr, false);
    }

    [[nodiscard]] Value *_translate_member_or_access_expr(XIRBuilder &b, const Expression *expr, bool load_lval) noexcept {
        luisa::fixed_vector<Value *, 16u> args;
        auto base = _collect_access_indices(b, expr, args);
        if (base->is_lvalue()) {
            std::reverse(args.begin(), args.end());
            auto elem = b.gep(expr->type(), base, args);
            return load_lval ? static_cast<Value *>(b.load(expr->type(), elem)) :
                               static_cast<Value *>(elem);
        }
        args.emplace_back(base);
        std::reverse(args.begin(), args.end());
        return b.call(expr->type(), ArithmeticOp::EXTRACT, args);
    }

    [[nodiscard]] Value *_translate_member_expr(XIRBuilder &b, const MemberExpr *expr, bool load_lval) noexcept {
        if (expr->is_swizzle()) {
            if (expr->swizzle_size() == 1u) {
                auto v = _translate_expression(b, expr->self(), load_lval);
                auto index = _translate_constant_access_index(expr->swizzle_index(0));
                if (v->is_lvalue()) {
                    LUISA_ASSERT(!load_lval, "Unexpected lvalue swizzle.");
                    return b.gep(expr->type(), v, {index});
                }
                return b.call(expr->type(), ArithmeticOp::EXTRACT, {v, index});
            }
            luisa::fixed_vector<Value *, 5u> args;
            auto v = _translate_expression(b, expr->self(), true);
            args.emplace_back(v);
            for (auto i = 0u; i < expr->swizzle_size(); i++) {
                auto index = expr->swizzle_index(i);
                args.emplace_back(_translate_constant_access_index(index));
            }
            return b.call(expr->type(), ArithmeticOp::SHUFFLE, args);
        }
        return _translate_member_or_access_expr(b, expr, load_lval);
    }

    [[nodiscard]] Value *_translate_typed_literal(TypedLiteral key) noexcept {
        auto [iter, just_inserted] = _generated_literals.try_emplace(key, nullptr);
        if (just_inserted) {
            iter->second = luisa::visit(
                [this, t = key.type]<typename T>(T v) noexcept {
                    LUISA_ASSERT(t == Type::of<T>(), "Literal type mismatch.");
                    return _module->create_constant(t, &v);
                },
                key.value);
        }
        return iter->second;
    }

    [[nodiscard]] Value *_translate_literal_expr(const LiteralExpr *expr) noexcept {
        auto key = TypedLiteral{expr->type(), expr->value()};
        return _translate_typed_literal(key);
    }

    [[nodiscard]] Value *_translate_builtin_variable(Variable ast_var) noexcept {
        LUISA_ASSERT(ast_var.is_builtin(), "Unresolved variable reference.");
        auto r = [m = this->_module.get(), tag = ast_var.tag()]() noexcept -> SpecialRegister * {
            switch (tag) {
                case Variable::Tag::THREAD_ID: return m->create_thread_id();
                case Variable::Tag::BLOCK_ID: return m->create_block_id();
                case Variable::Tag::DISPATCH_ID: return m->create_dispatch_id();
                case Variable::Tag::DISPATCH_SIZE: return m->create_dispatch_size();
                case Variable::Tag::KERNEL_ID: return m->create_kernel_id();
                case Variable::Tag::WARP_LANE_COUNT: return m->create_warp_size();
                case Variable::Tag::WARP_LANE_ID: return m->create_warp_lane_id();
                case Variable::Tag::OBJECT_ID: return m->create_object_id();
                default: break;
            }
            LUISA_ERROR_WITH_LOCATION("Unexpected variable type.");
        }();
        LUISA_ASSERT(r->type() == ast_var.type(), "Special register {} type mismatch: {} vs {}.",
                     xir::to_string(r->derived_special_register_tag()),
                     r->type()->description(), ast_var.type()->description());
        return r;
    }

    [[nodiscard]] Value *_translate_ref_expr(XIRBuilder &b, const RefExpr *expr, bool load_lval) noexcept {
        auto ast_var = expr->variable();
        LUISA_ASSERT(ast_var.type() == expr->type(), "Variable type mismatch.");
        if (auto iter = _current.variables.find(ast_var); iter != _current.variables.end()) {
            auto var = iter->second;
            return load_lval && var->is_lvalue() ? b.load(expr->type(), var) : var;
        }
        return _translate_builtin_variable(ast_var);
    }

    [[nodiscard]] Value *_translate_constant_expr(const ConstantExpr *expr) noexcept {
        auto c = expr->data();
        auto [iter, just_inserted] = _generated_constants.try_emplace(c, nullptr);
        if (just_inserted) {
            iter->second = _module->create_constant(c.type(), c.raw());
        }
        return iter->second;
    }

    [[nodiscard]] Value *_translate_zero_or_one(const Type *type, int value) noexcept {

        // zero or one scalar
#define LUISA_AST2XIR_ZERO_ONE_SCALAR(T)    \
    if (type == Type::of<T>()) {            \
        return _translate_typed_literal(    \
            {type, static_cast<T>(value)}); \
    }
        LUISA_AST2XIR_ZERO_ONE_SCALAR(bool)
        LUISA_AST2XIR_ZERO_ONE_SCALAR(byte)
        LUISA_AST2XIR_ZERO_ONE_SCALAR(ubyte)
        LUISA_AST2XIR_ZERO_ONE_SCALAR(short)
        LUISA_AST2XIR_ZERO_ONE_SCALAR(ushort)
        LUISA_AST2XIR_ZERO_ONE_SCALAR(int)
        LUISA_AST2XIR_ZERO_ONE_SCALAR(uint)
        LUISA_AST2XIR_ZERO_ONE_SCALAR(slong)
        LUISA_AST2XIR_ZERO_ONE_SCALAR(ulong)
        LUISA_AST2XIR_ZERO_ONE_SCALAR(half)
        LUISA_AST2XIR_ZERO_ONE_SCALAR(float)
        LUISA_AST2XIR_ZERO_ONE_SCALAR(double)
#undef LUISA_AST2XIR_ZERO_ONE_SCALAR

        // zero or one vector
#define LUISA_AST2XIR_ZERO_ONE_VECTOR_N(T, N)                   \
    if (type == Type::of<T##N>()) {                             \
        return _translate_typed_literal(                        \
            {type, luisa::make_##T##N(static_cast<T>(value))}); \
    }
#define LUISA_AST2XIR_ZERO_ONE_VECTOR(T)  \
    LUISA_AST2XIR_ZERO_ONE_VECTOR_N(T, 2) \
    LUISA_AST2XIR_ZERO_ONE_VECTOR_N(T, 3) \
    LUISA_AST2XIR_ZERO_ONE_VECTOR_N(T, 4)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(bool)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(byte)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(ubyte)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(short)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(ushort)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(int)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(uint)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(slong)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(ulong)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(half)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(float)
        LUISA_AST2XIR_ZERO_ONE_VECTOR(double)
#undef LUISA_AST2XIR_ZERO_ONE_VECTOR
#undef LUISA_AST2XIR_ZERO_ONE_VECTOR_N

        // zero or one matrix
#define LUISA_AST2XIR_ZERO_ONE_MATRIX(N)                                    \
    if (type == Type::of<luisa::float##N##x##N>()) {                        \
        return _translate_typed_literal(                                    \
            {type, luisa::make_float##N##x##N(static_cast<float>(value))}); \
    }
        LUISA_AST2XIR_ZERO_ONE_MATRIX(2)
        LUISA_AST2XIR_ZERO_ONE_MATRIX(3)
        LUISA_AST2XIR_ZERO_ONE_MATRIX(4)
#undef LUISA_AST2XIR_ZERO_ONE_MATRIX

        // fall back to generic zero constant
        if (value == 0) {
            auto iter = _generated_zero_constants.try_emplace(type, nullptr).first;
            if (iter->second == nullptr) { iter->second = _module->create_constant_zero(type); }
            return iter->second;
        }
        if (value == 1) {
            auto iter = _generated_one_constants.try_emplace(type, nullptr).first;
            if (iter->second == nullptr) { iter->second = _module->create_constant_one(type); }
            return iter->second;
        }
        LUISA_ERROR_WITH_LOCATION("Unexpected zero or one constant.");
    }

    [[nodiscard]] Value *_translate_call_expr(XIRBuilder &b, const CallExpr *expr) noexcept {
        if (expr->is_external()) {
            auto ast = expr->external();
            auto f = add_external_function(*ast);
            LUISA_NOT_IMPLEMENTED();
        }
        if (expr->is_custom()) {
            auto ast = expr->custom();
            auto f = add_function(ast);
            LUISA_ASSERT(f->type() == expr->type(), "Function return type mismatch.");
            auto ast_args = expr->arguments();
            luisa::fixed_vector<Value *, 16u> args;
            args.reserve(expr->arguments().size());
            for (auto i = 0u; i < ast_args.size(); i++) {
                auto by_ref = ast.arguments()[i].is_reference();
                auto arg = _translate_expression(b, ast_args[i], !by_ref);
                args.emplace_back(arg);
            }
            return b.call(f->type(), f, args);
        }
        auto alu_call = [&](ArithmeticOp target_op) noexcept {
            luisa::fixed_vector<Value *, 16u> args;
            args.reserve(expr->arguments().size());
            for (auto ast_arg : expr->arguments()) {
                auto arg = _translate_expression(b, ast_arg, true);
                args.emplace_back(arg);
            }
            return b.call(expr->type(), target_op, args);
        };
        auto cta_call = [&](ThreadGroupOp target_op) noexcept {
            luisa::fixed_vector<Value *, 16u> args;
            args.reserve(expr->arguments().size());
            for (auto ast_arg : expr->arguments()) {
                auto arg = _translate_expression(b, ast_arg, true);
                args.emplace_back(arg);
            }
            return b.call(expr->type(), target_op, args);
        };
        auto resource_call = [&]<typename ResourceOp>(ResourceOp target_op) noexcept {
            static_assert(std::is_same_v<ResourceOp, ResourceQueryOp> ||
                          std::is_same_v<ResourceOp, ResourceReadOp> ||
                          std::is_same_v<ResourceOp, ResourceWriteOp>);
            LUISA_ASSERT(!expr->arguments().empty(), "Resource call requires at least one argument.");
            luisa::fixed_vector<Value *, 16u> args;
            args.reserve(expr->arguments().size());
            auto base = _translate_expression(b, expr->arguments()[0], false);
            args.emplace_back(base);
            auto other = expr->arguments().subspan(1);
            for (auto ast_arg : other) {
                auto arg = _translate_expression(b, ast_arg, true);
                args.emplace_back(arg);
            }
            if constexpr (std::is_same_v<ResourceOp, ResourceWriteOp>) {
                return b.call(target_op, args);
            } else {
                return b.call(expr->type(), target_op, args);
            }
        };
        auto rq_call = [&]<typename T>(T target_op) noexcept {
            static_assert(std::is_same_v<T, RayQueryObjectReadOp> || std::is_same_v<T, RayQueryObjectWriteOp>);
            LUISA_ASSERT(!expr->arguments().empty(), "RayQuery call requires at least one argument.");
            luisa::fixed_vector<Value *, 16u> args;
            args.reserve(expr->arguments().size());
            auto base = _translate_expression(b, expr->arguments()[0], false);
            args.emplace_back(base);
            auto other = expr->arguments().subspan(1);
            for (auto ast_arg : other) {
                auto arg = _translate_expression(b, ast_arg, true);
                args.emplace_back(arg);
            }
            if constexpr (std::is_same_v<T, RayQueryObjectWriteOp>) {
                return b.call(target_op, args);
            } else {
                return b.call(expr->type(), target_op, args);
            }
        };
        auto atomic_call = [&](AtomicOp target_op) noexcept {
            LUISA_ASSERT(!expr->arguments().empty(), "Atomic call requires at least one argument.");
            auto base = _translate_expression(b, expr->arguments()[0], false);
            auto other = expr->arguments().subspan(1);
            auto value_count = xir::atomic_op_value_count(target_op);
            LUISA_ASSERT(other.size() >= value_count, "Invalid number of arguments for atomic call.");
            luisa::fixed_vector<Value *, 16u> args;
            args.reserve(other.size());
            for (auto ast_arg : other) {
                auto arg = _translate_expression(b, ast_arg, true);
                args.emplace_back(arg);
            }
            auto indices = luisa::span{args}.subspan(0u, args.size() - value_count);
            auto values = luisa::span{args}.subspan(args.size() - value_count, value_count);
            return b.call(expr->type(), target_op, base, indices, values);
        };
        auto texture_dim = [&]() noexcept {
            LUISA_ASSERT(!expr->arguments().empty(), "Texture dimension call requires at least one argument.");
            auto tex = expr->arguments()[0];
            LUISA_ASSERT(tex->tag() == Expression::Tag::REF, "Texture dimension call requires a texture reference.");
            auto ref = static_cast<const RefExpr *>(tex);
            auto ast_var = ref->variable();
            LUISA_ASSERT(ast_var.tag() == Variable::Tag::TEXTURE, "Texture dimension call requires a texture reference.");
            auto type = ast_var.type();
            LUISA_ASSERT(type->is_texture(), "Texture dimension call requires a texture reference.");
            return type->dimension();
        };
        auto make_vector_call = [&](const Type *elem_type, int dim) noexcept -> Value * {
            LUISA_ASSERT(dim == 2 || dim == 3 || dim == 4, "Vector call only supports 2, 3 or 4 dimension.");
            auto ast_args = expr->arguments();
            if (ast_args.size() == 1) {
                auto arg = _translate_expression(b, ast_args[0], true);
                return _type_cast_if_necessary(b, expr->type(), arg);
            }
            luisa::fixed_vector<Value *, 4u> args;
            for (auto ast_arg : ast_args) {
                auto arg = _translate_expression(b, ast_arg, true);
                if (arg->type()->is_scalar()) {
                    args.emplace_back(_type_cast_if_necessary(b, elem_type, arg));
                } else {
                    LUISA_ASSERT(arg->type()->is_vector(), "Vector call argument type mismatch.");
                    auto arg_elem_type = arg->type()->element();
                    for (auto i = 0u; i < arg->type()->dimension(); i++) {
                        auto idx = _translate_constant_access_index(i);
                        auto elem = b.call(arg_elem_type, ArithmeticOp::EXTRACT, {arg, idx});
                        args.emplace_back(_type_cast_if_necessary(b, elem_type, elem));
                    }
                }
            }
            LUISA_ASSERT(args.size() == dim, "Vector call requires {} arguments.", dim);
            return b.call(expr->type(), ArithmeticOp::AGGREGATE, args);
        };
        auto make_matrix_call = [&](const Type *elem_type, int dim) noexcept {
            LUISA_ASSERT(elem_type == Type::of<float>(), "Matrix call only supports float element type.");
            LUISA_ASSERT(dim == 2 || dim == 3 || dim == 4, "Matrix call only supports 2x2, 3x3 or 4x4 matrix.");
            auto ast_args = expr->arguments();
            LUISA_ASSERT(ast_args.size() == dim, "Matrix call requires {} arguments.", dim);
            luisa::fixed_vector<Value *, 4u> args;
            auto col_type = Type::vector(elem_type, dim);
            for (auto ast_arg : ast_args) {
                LUISA_ASSERT(ast_arg->type() == col_type, "Matrix call argument type mismatch.");
                auto arg = _translate_expression(b, ast_arg, true);
                args.emplace_back(arg);
            }
            return b.call(expr->type(), ArithmeticOp::AGGREGATE, args);
        };
        // builtin function
        switch (expr->op()) {
            case CallOp::CUSTOM: LUISA_ERROR_WITH_LOCATION("Unexpected custom call operation.");
            case CallOp::EXTERNAL: LUISA_ERROR_WITH_LOCATION("Unexpected external call operation.");
            case CallOp::ALL: return alu_call(ArithmeticOp::ALL);
            case CallOp::ANY: return alu_call(ArithmeticOp::ANY);
            case CallOp::SELECT: return alu_call(ArithmeticOp::SELECT);
            case CallOp::CLAMP: return alu_call(ArithmeticOp::CLAMP);
            case CallOp::SATURATE: return alu_call(ArithmeticOp::SATURATE);
            case CallOp::LERP: return alu_call(ArithmeticOp::LERP);
            case CallOp::SMOOTHSTEP: return alu_call(ArithmeticOp::SMOOTHSTEP);
            case CallOp::STEP: return alu_call(ArithmeticOp::STEP);
            case CallOp::ABS: return alu_call(ArithmeticOp::ABS);
            case CallOp::MIN: return alu_call(ArithmeticOp::MIN);
            case CallOp::MAX: return alu_call(ArithmeticOp::MAX);
            case CallOp::CLZ: return alu_call(ArithmeticOp::CLZ);
            case CallOp::CTZ: return alu_call(ArithmeticOp::CTZ);
            case CallOp::POPCOUNT: return alu_call(ArithmeticOp::POPCOUNT);
            case CallOp::REVERSE: return alu_call(ArithmeticOp::REVERSE);
            case CallOp::ISINF: return alu_call(ArithmeticOp::ISINF);
            case CallOp::ISNAN: return alu_call(ArithmeticOp::ISNAN);
            case CallOp::ACOS: return alu_call(ArithmeticOp::ACOS);
            case CallOp::ACOSH: return alu_call(ArithmeticOp::ACOSH);
            case CallOp::ASIN: return alu_call(ArithmeticOp::ASIN);
            case CallOp::ASINH: return alu_call(ArithmeticOp::ASINH);
            case CallOp::ATAN: return alu_call(ArithmeticOp::ATAN);
            case CallOp::ATAN2: return alu_call(ArithmeticOp::ATAN2);
            case CallOp::ATANH: return alu_call(ArithmeticOp::ATANH);
            case CallOp::COS: return alu_call(ArithmeticOp::COS);
            case CallOp::COSH: return alu_call(ArithmeticOp::COSH);
            case CallOp::SIN: return alu_call(ArithmeticOp::SIN);
            case CallOp::SINH: return alu_call(ArithmeticOp::SINH);
            case CallOp::TAN: return alu_call(ArithmeticOp::TAN);
            case CallOp::TANH: return alu_call(ArithmeticOp::TANH);
            case CallOp::EXP: return alu_call(ArithmeticOp::EXP);
            case CallOp::EXP2: return alu_call(ArithmeticOp::EXP2);
            case CallOp::EXP10: return alu_call(ArithmeticOp::EXP10);
            case CallOp::LOG: return alu_call(ArithmeticOp::LOG);
            case CallOp::LOG2: return alu_call(ArithmeticOp::LOG2);
            case CallOp::LOG10: return alu_call(ArithmeticOp::LOG10);
            case CallOp::POW: return alu_call(ArithmeticOp::POW);
            case CallOp::SQRT: return alu_call(ArithmeticOp::SQRT);
            case CallOp::RSQRT: return alu_call(ArithmeticOp::RSQRT);
            case CallOp::CEIL: return alu_call(ArithmeticOp::CEIL);
            case CallOp::FLOOR: return alu_call(ArithmeticOp::FLOOR);
            case CallOp::FRACT: return alu_call(ArithmeticOp::FRACT);
            case CallOp::TRUNC: return alu_call(ArithmeticOp::TRUNC);
            case CallOp::ROUND: return alu_call(ArithmeticOp::ROUND);
            case CallOp::FMA: return alu_call(ArithmeticOp::FMA);
            case CallOp::COPYSIGN: return alu_call(ArithmeticOp::COPYSIGN);
            case CallOp::CROSS: return alu_call(ArithmeticOp::CROSS);
            case CallOp::DOT: return alu_call(ArithmeticOp::DOT);
            case CallOp::LENGTH: return alu_call(ArithmeticOp::LENGTH);
            case CallOp::LENGTH_SQUARED: return alu_call(ArithmeticOp::LENGTH_SQUARED);
            case CallOp::NORMALIZE: return alu_call(ArithmeticOp::NORMALIZE);
            case CallOp::FACEFORWARD: return alu_call(ArithmeticOp::FACEFORWARD);
            case CallOp::REFLECT: return alu_call(ArithmeticOp::REFLECT);
            case CallOp::REDUCE_SUM: return alu_call(ArithmeticOp::REDUCE_SUM);
            case CallOp::REDUCE_PRODUCT: return alu_call(ArithmeticOp::REDUCE_PRODUCT);
            case CallOp::REDUCE_MIN: return alu_call(ArithmeticOp::REDUCE_MIN);
            case CallOp::REDUCE_MAX: return alu_call(ArithmeticOp::REDUCE_MAX);
            case CallOp::OUTER_PRODUCT: return alu_call(ArithmeticOp::OUTER_PRODUCT);
            case CallOp::MATRIX_COMPONENT_WISE_MULTIPLICATION: return alu_call(ArithmeticOp::MATRIX_COMP_MUL);
            case CallOp::DETERMINANT: return alu_call(ArithmeticOp::MATRIX_DETERMINANT);
            case CallOp::TRANSPOSE: return alu_call(ArithmeticOp::MATRIX_TRANSPOSE);
            case CallOp::INVERSE: return alu_call(ArithmeticOp::MATRIX_INVERSE);
            case CallOp::ATOMIC_EXCHANGE: return atomic_call(AtomicOp::EXCHANGE);
            case CallOp::ATOMIC_COMPARE_EXCHANGE: return atomic_call(AtomicOp::COMPARE_EXCHANGE);
            case CallOp::ATOMIC_FETCH_ADD: return atomic_call(AtomicOp::FETCH_ADD);
            case CallOp::ATOMIC_FETCH_SUB: return atomic_call(AtomicOp::FETCH_SUB);
            case CallOp::ATOMIC_FETCH_AND: return atomic_call(AtomicOp::FETCH_AND);
            case CallOp::ATOMIC_FETCH_OR: return atomic_call(AtomicOp::FETCH_OR);
            case CallOp::ATOMIC_FETCH_XOR: return atomic_call(AtomicOp::FETCH_XOR);
            case CallOp::ATOMIC_FETCH_MIN: return atomic_call(AtomicOp::FETCH_MIN);
            case CallOp::ATOMIC_FETCH_MAX: return atomic_call(AtomicOp::FETCH_MAX);
            case CallOp::ADDRESS_OF: LUISA_ERROR_WITH_LOCATION("Removed address_of operation.");
            case CallOp::BUFFER_READ: return resource_call(ResourceReadOp::BUFFER_READ);
            case CallOp::BUFFER_WRITE: return resource_call(ResourceWriteOp::BUFFER_WRITE);
            case CallOp::BUFFER_SIZE: return resource_call(ResourceQueryOp::BUFFER_SIZE);
            case CallOp::BUFFER_ADDRESS: return resource_call(ResourceQueryOp::BUFFER_DEVICE_ADDRESS);
            case CallOp::BYTE_BUFFER_READ: return resource_call(ResourceReadOp::BYTE_BUFFER_READ);
            case CallOp::BYTE_BUFFER_WRITE: return resource_call(ResourceWriteOp::BYTE_BUFFER_WRITE);
            case CallOp::BYTE_BUFFER_SIZE: return resource_call(ResourceQueryOp::BYTE_BUFFER_SIZE);
            case CallOp::TEXTURE_READ: return resource_call(texture_dim() == 2u ? ResourceReadOp::TEXTURE2D_READ : ResourceReadOp::TEXTURE3D_READ);
            case CallOp::TEXTURE_WRITE: return resource_call(texture_dim() == 2u ? ResourceWriteOp::TEXTURE2D_WRITE : ResourceWriteOp::TEXTURE3D_WRITE);
            case CallOp::TEXTURE_SIZE: return resource_call(texture_dim() == 2u ? ResourceQueryOp::TEXTURE2D_SIZE : ResourceQueryOp::TEXTURE3D_SIZE);
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE);
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL);
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD);
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL);
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE);
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL);
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD);
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL);
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_SAMPLER);
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_LEVEL_SAMPLER);
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_SAMPLER);
            case CallOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE2D_SAMPLE_GRAD_LEVEL_SAMPLER);
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_SAMPLER);
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_LEVEL_SAMPLER);
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_SAMPLER);
            case CallOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE3D_SAMPLE_GRAD_LEVEL_SAMPLER);
            case CallOp::BINDLESS_TEXTURE2D_READ: return resource_call(ResourceReadOp::BINDLESS_TEXTURE2D_READ);
            case CallOp::BINDLESS_TEXTURE3D_READ: return resource_call(ResourceReadOp::BINDLESS_TEXTURE3D_READ);
            case CallOp::BINDLESS_TEXTURE2D_READ_LEVEL: return resource_call(ResourceReadOp::BINDLESS_TEXTURE2D_READ_LEVEL);
            case CallOp::BINDLESS_TEXTURE3D_READ_LEVEL: return resource_call(ResourceReadOp::BINDLESS_TEXTURE3D_READ_LEVEL);
            case CallOp::BINDLESS_TEXTURE2D_SIZE: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE);
            case CallOp::BINDLESS_TEXTURE3D_SIZE: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE);
            case CallOp::BINDLESS_TEXTURE2D_SIZE_LEVEL: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE2D_SIZE_LEVEL);
            case CallOp::BINDLESS_TEXTURE3D_SIZE_LEVEL: return resource_call(ResourceQueryOp::BINDLESS_TEXTURE3D_SIZE_LEVEL);
            case CallOp::BINDLESS_BUFFER_READ: return resource_call(ResourceReadOp::BINDLESS_BUFFER_READ);
            case CallOp::BINDLESS_BUFFER_WRITE: return resource_call(ResourceWriteOp::BINDLESS_BUFFER_WRITE);
            case CallOp::BINDLESS_BYTE_BUFFER_READ: return resource_call(ResourceReadOp::BINDLESS_BYTE_BUFFER_READ);
            case CallOp::BINDLESS_BUFFER_SIZE: return resource_call(ResourceQueryOp::BINDLESS_BUFFER_SIZE);
            case CallOp::BINDLESS_BUFFER_TYPE: LUISA_ERROR_WITH_LOCATION("Removed bindless_buffer_type operation.");
            case CallOp::BINDLESS_BUFFER_ADDRESS: return resource_call(ResourceQueryOp::BINDLESS_BUFFER_DEVICE_ADDRESS);
            case CallOp::MAKE_BOOL2: return make_vector_call(Type::of<bool>(), 2);
            case CallOp::MAKE_BOOL3: return make_vector_call(Type::of<bool>(), 3);
            case CallOp::MAKE_BOOL4: return make_vector_call(Type::of<bool>(), 4);
            case CallOp::MAKE_INT2: return make_vector_call(Type::of<int>(), 2);
            case CallOp::MAKE_INT3: return make_vector_call(Type::of<int>(), 3);
            case CallOp::MAKE_INT4: return make_vector_call(Type::of<int>(), 4);
            case CallOp::MAKE_UINT2: return make_vector_call(Type::of<uint>(), 2);
            case CallOp::MAKE_UINT3: return make_vector_call(Type::of<uint>(), 3);
            case CallOp::MAKE_UINT4: return make_vector_call(Type::of<uint>(), 4);
            case CallOp::MAKE_FLOAT2: return make_vector_call(Type::of<float>(), 2);
            case CallOp::MAKE_FLOAT3: return make_vector_call(Type::of<float>(), 3);
            case CallOp::MAKE_FLOAT4: return make_vector_call(Type::of<float>(), 4);
            case CallOp::MAKE_SHORT2: return make_vector_call(Type::of<short>(), 2);
            case CallOp::MAKE_SHORT3: return make_vector_call(Type::of<short>(), 3);
            case CallOp::MAKE_SHORT4: return make_vector_call(Type::of<short>(), 4);
            case CallOp::MAKE_USHORT2: return make_vector_call(Type::of<ushort>(), 2);
            case CallOp::MAKE_USHORT3: return make_vector_call(Type::of<ushort>(), 3);
            case CallOp::MAKE_USHORT4: return make_vector_call(Type::of<ushort>(), 4);
            case CallOp::MAKE_LONG2: return make_vector_call(Type::of<long>(), 2);
            case CallOp::MAKE_LONG3: return make_vector_call(Type::of<long>(), 3);
            case CallOp::MAKE_LONG4: return make_vector_call(Type::of<long>(), 4);
            case CallOp::MAKE_ULONG2: return make_vector_call(Type::of<ulong>(), 2);
            case CallOp::MAKE_ULONG3: return make_vector_call(Type::of<ulong>(), 3);
            case CallOp::MAKE_ULONG4: return make_vector_call(Type::of<ulong>(), 4);
            case CallOp::MAKE_HALF2: return make_vector_call(Type::of<half>(), 2);
            case CallOp::MAKE_HALF3: return make_vector_call(Type::of<half>(), 3);
            case CallOp::MAKE_HALF4: return make_vector_call(Type::of<half>(), 4);
            case CallOp::MAKE_DOUBLE2: return make_vector_call(Type::of<double>(), 2);
            case CallOp::MAKE_DOUBLE3: return make_vector_call(Type::of<double>(), 3);
            case CallOp::MAKE_DOUBLE4: return make_vector_call(Type::of<double>(), 4);
            case CallOp::MAKE_BYTE2: return make_vector_call(Type::of<byte>(), 2);
            case CallOp::MAKE_BYTE3: return make_vector_call(Type::of<byte>(), 3);
            case CallOp::MAKE_BYTE4: return make_vector_call(Type::of<byte>(), 4);
            case CallOp::MAKE_UBYTE2: return make_vector_call(Type::of<ubyte>(), 2);
            case CallOp::MAKE_UBYTE3: return make_vector_call(Type::of<ubyte>(), 3);
            case CallOp::MAKE_UBYTE4: return make_vector_call(Type::of<ubyte>(), 4);
            case CallOp::MAKE_FLOAT2X2: return make_matrix_call(Type::of<float>(), 2);
            case CallOp::MAKE_FLOAT3X3: return make_matrix_call(Type::of<float>(), 3);
            case CallOp::MAKE_FLOAT4X4: return make_matrix_call(Type::of<float>(), 4);
            case CallOp::ASSERT: {
                LUISA_ASSERT(!expr->arguments().empty(), "Assert requires at least one argument.");
                auto cond = _translate_expression(b, expr->arguments()[0], true);
                luisa::string_view message;
                if (expr->arguments().size() >= 2u) {
                    auto ast_msg_id = expr->arguments()[1];
                    LUISA_ASSERT(ast_msg_id->tag() == Expression::Tag::STRING_ID, "Assert message must be a string.");
                    auto msg_id = static_cast<const StringIDExpr *>(ast_msg_id);
                    message = msg_id->data();
                }
                return b.assert_(cond, message);
            }
            case CallOp::ASSUME: {
                LUISA_ASSERT(!expr->arguments().empty(), "Assume requires at least one argument.");
                auto cond = _translate_expression(b, expr->arguments()[0], true);
                luisa::string_view message;
                if (expr->arguments().size() >= 2u) {
                    auto ast_msg_id = expr->arguments()[1];
                    LUISA_ASSERT(ast_msg_id->tag() == Expression::Tag::STRING_ID, "Assume message must be a string.");
                    auto msg_id = static_cast<const StringIDExpr *>(ast_msg_id);
                    message = msg_id->data();
                }
                return b.assume_(cond, message);
            }
            case CallOp::UNREACHABLE: {
                luisa::string_view message;
                if (!expr->arguments().empty()) {
                    auto ast_msg_id = expr->arguments()[0];
                    LUISA_ASSERT(ast_msg_id->tag() == Expression::Tag::STRING_ID, "Unreachable message must be a string.");
                    auto msg_id = static_cast<const StringIDExpr *>(ast_msg_id);
                    message = msg_id->data();
                }
                return b.unreachable_(message);
            }
            case CallOp::RASTER_DISCARD: return b.raster_discard();
            case CallOp::ZERO: return _translate_zero_or_one(expr->type(), 0);
            case CallOp::ONE: return _translate_zero_or_one(expr->type(), 1);
            case CallOp::PACK: LUISA_NOT_IMPLEMENTED();
            case CallOp::UNPACK: LUISA_NOT_IMPLEMENTED();
            case CallOp::REQUIRES_GRADIENT: LUISA_NOT_IMPLEMENTED();
            case CallOp::GRADIENT: LUISA_NOT_IMPLEMENTED();
            case CallOp::GRADIENT_MARKER: LUISA_NOT_IMPLEMENTED();
            case CallOp::ACCUMULATE_GRADIENT: LUISA_NOT_IMPLEMENTED();
            case CallOp::BACKWARD: LUISA_NOT_IMPLEMENTED();
            case CallOp::DETACH: LUISA_NOT_IMPLEMENTED();
            case CallOp::RAY_TRACING_INSTANCE_TRANSFORM: return resource_call(ResourceQueryOp::RAY_TRACING_INSTANCE_TRANSFORM);
            case CallOp::RAY_TRACING_INSTANCE_USER_ID: return resource_call(ResourceQueryOp::RAY_TRACING_INSTANCE_USER_ID);
            case CallOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK: return resource_call(ResourceQueryOp::RAY_TRACING_INSTANCE_VISIBILITY_MASK);
            case CallOp::RAY_TRACING_SET_INSTANCE_TRANSFORM: return resource_call(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_TRANSFORM);
            case CallOp::RAY_TRACING_SET_INSTANCE_VISIBILITY: return resource_call(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_VISIBILITY_MASK);
            case CallOp::RAY_TRACING_SET_INSTANCE_OPACITY: return resource_call(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_OPACITY);
            case CallOp::RAY_TRACING_SET_INSTANCE_USER_ID: return resource_call(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_USER_ID);
            case CallOp::RAY_TRACING_TRACE_CLOSEST: return resource_call(ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST);
            case CallOp::RAY_TRACING_TRACE_ANY: return resource_call(ResourceQueryOp::RAY_TRACING_TRACE_ANY);
            case CallOp::RAY_TRACING_QUERY_ALL: return resource_call(ResourceQueryOp::RAY_TRACING_QUERY_ALL);
            case CallOp::RAY_TRACING_QUERY_ANY: return resource_call(ResourceQueryOp::RAY_TRACING_QUERY_ANY);
            case CallOp::RAY_TRACING_INSTANCE_MOTION_MATRIX: return resource_call(ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_MATRIX);
            case CallOp::RAY_TRACING_INSTANCE_MOTION_SRT: return resource_call(ResourceQueryOp::RAY_TRACING_INSTANCE_MOTION_SRT);
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX: return resource_call(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_MATRIX);
            case CallOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT: return resource_call(ResourceWriteOp::RAY_TRACING_SET_INSTANCE_MOTION_SRT);
            case CallOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR: return resource_call(ResourceQueryOp::RAY_TRACING_TRACE_CLOSEST_MOTION_BLUR);
            case CallOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR: return resource_call(ResourceQueryOp::RAY_TRACING_TRACE_ANY_MOTION_BLUR);
            case CallOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR: return resource_call(ResourceQueryOp::RAY_TRACING_QUERY_ALL_MOTION_BLUR);
            case CallOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR: return resource_call(ResourceQueryOp::RAY_TRACING_QUERY_ANY_MOTION_BLUR);
            case CallOp::RAY_QUERY_WORLD_SPACE_RAY: return rq_call(RayQueryObjectReadOp::RAY_QUERY_OBJECT_WORLD_SPACE_RAY);
            case CallOp::RAY_QUERY_PROCEDURAL_CANDIDATE_HIT: return rq_call(RayQueryObjectReadOp::RAY_QUERY_OBJECT_PROCEDURAL_CANDIDATE_HIT);
            case CallOp::RAY_QUERY_TRIANGLE_CANDIDATE_HIT: return rq_call(RayQueryObjectReadOp::RAY_QUERY_OBJECT_TRIANGLE_CANDIDATE_HIT);
            case CallOp::RAY_QUERY_COMMITTED_HIT: return rq_call(RayQueryObjectReadOp::RAY_QUERY_OBJECT_COMMITTED_HIT);
            case CallOp::RAY_QUERY_COMMIT_TRIANGLE: return rq_call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_TRIANGLE);
            case CallOp::RAY_QUERY_COMMIT_PROCEDURAL: return rq_call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_COMMIT_PROCEDURAL);
            case CallOp::RAY_QUERY_TERMINATE: return rq_call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_TERMINATE);
            case CallOp::RAY_QUERY_PROCEED: {
                b.call(RayQueryObjectWriteOp::RAY_QUERY_OBJECT_PROCEED, {});
                return rq_call(RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TERMINATED);
            }
            case CallOp::RAY_QUERY_IS_TRIANGLE_CANDIDATE: return rq_call(RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_TRIANGLE_CANDIDATE);
            case CallOp::RAY_QUERY_IS_PROCEDURAL_CANDIDATE: return rq_call(RayQueryObjectReadOp::RAY_QUERY_OBJECT_IS_PROCEDURAL_CANDIDATE);
            case CallOp::DDX: return cta_call(ThreadGroupOp::RASTER_QUAD_DDX);
            case CallOp::DDY: return cta_call(ThreadGroupOp::RASTER_QUAD_DDY);
            case CallOp::SHADER_EXECUTION_REORDER: return cta_call(ThreadGroupOp::SHADER_EXECUTION_REORDER);
            case CallOp::SYNCHRONIZE_BLOCK: return cta_call(ThreadGroupOp::SYNCHRONIZE_BLOCK);
            case CallOp::WARP_IS_FIRST_ACTIVE_LANE: return cta_call(ThreadGroupOp::WARP_IS_FIRST_ACTIVE_LANE);
            case CallOp::WARP_FIRST_ACTIVE_LANE: return cta_call(ThreadGroupOp::WARP_FIRST_ACTIVE_LANE);
            case CallOp::WARP_ACTIVE_ALL_EQUAL: return cta_call(ThreadGroupOp::WARP_ACTIVE_ALL_EQUAL);
            case CallOp::WARP_ACTIVE_BIT_AND: return cta_call(ThreadGroupOp::WARP_ACTIVE_BIT_AND);
            case CallOp::WARP_ACTIVE_BIT_OR: return cta_call(ThreadGroupOp::WARP_ACTIVE_BIT_OR);
            case CallOp::WARP_ACTIVE_BIT_XOR: return cta_call(ThreadGroupOp::WARP_ACTIVE_BIT_XOR);
            case CallOp::WARP_ACTIVE_COUNT_BITS: return cta_call(ThreadGroupOp::WARP_ACTIVE_COUNT_BITS);
            case CallOp::WARP_ACTIVE_MAX: return cta_call(ThreadGroupOp::WARP_ACTIVE_MAX);
            case CallOp::WARP_ACTIVE_MIN: return cta_call(ThreadGroupOp::WARP_ACTIVE_MIN);
            case CallOp::WARP_ACTIVE_PRODUCT: return cta_call(ThreadGroupOp::WARP_ACTIVE_PRODUCT);
            case CallOp::WARP_ACTIVE_SUM: return cta_call(ThreadGroupOp::WARP_ACTIVE_SUM);
            case CallOp::WARP_ACTIVE_ALL: return cta_call(ThreadGroupOp::WARP_ACTIVE_ALL);
            case CallOp::WARP_ACTIVE_ANY: return cta_call(ThreadGroupOp::WARP_ACTIVE_ANY);
            case CallOp::WARP_ACTIVE_BIT_MASK: return cta_call(ThreadGroupOp::WARP_ACTIVE_BIT_MASK);
            case CallOp::WARP_PREFIX_COUNT_BITS: return cta_call(ThreadGroupOp::WARP_PREFIX_COUNT_BITS);
            case CallOp::WARP_PREFIX_SUM: return cta_call(ThreadGroupOp::WARP_PREFIX_SUM);
            case CallOp::WARP_PREFIX_PRODUCT: return cta_call(ThreadGroupOp::WARP_PREFIX_PRODUCT);
            case CallOp::WARP_READ_LANE: return cta_call(ThreadGroupOp::WARP_READ_LANE);
            case CallOp::WARP_READ_FIRST_ACTIVE_LANE: return cta_call(ThreadGroupOp::WARP_READ_FIRST_ACTIVE_LANE);
            case CallOp::INDIRECT_SET_DISPATCH_KERNEL: return resource_call(ResourceWriteOp::INDIRECT_DISPATCH_SET_KERNEL);
            case CallOp::INDIRECT_SET_DISPATCH_COUNT: return resource_call(ResourceWriteOp::INDIRECT_DISPATCH_SET_COUNT);
            case CallOp::TEXTURE2D_SAMPLE: return resource_call(ResourceQueryOp::TEXTURE2D_SAMPLE);
            case CallOp::TEXTURE2D_SAMPLE_LEVEL: return resource_call(ResourceQueryOp::TEXTURE2D_SAMPLE_LEVEL);
            case CallOp::TEXTURE2D_SAMPLE_GRAD: return resource_call(ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD);
            case CallOp::TEXTURE2D_SAMPLE_GRAD_LEVEL: return resource_call(ResourceQueryOp::TEXTURE2D_SAMPLE_GRAD_LEVEL);
            case CallOp::TEXTURE3D_SAMPLE: return resource_call(ResourceQueryOp::TEXTURE3D_SAMPLE);
            case CallOp::TEXTURE3D_SAMPLE_LEVEL: return resource_call(ResourceQueryOp::TEXTURE3D_SAMPLE_LEVEL);
            case CallOp::TEXTURE3D_SAMPLE_GRAD: return resource_call(ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD);
            case CallOp::TEXTURE3D_SAMPLE_GRAD_LEVEL: return resource_call(ResourceQueryOp::TEXTURE3D_SAMPLE_GRAD_LEVEL);
            case CallOp::CLOCK: return b.clock();
        }
        LUISA_NOT_IMPLEMENTED();
    }

    [[nodiscard]] Value *_translate_cast_expr(XIRBuilder &b, const CastExpr *expr) noexcept {
        auto value = _translate_expression(b, expr->expression(), true);
        switch (expr->op()) {
            case compute::CastOp::STATIC: return _type_cast_if_necessary(b, expr->type(), value);
            case compute::CastOp::BITWISE: return b.bit_cast_if_necessary(expr->type(), value);
        }
        LUISA_ERROR_WITH_LOCATION("Unexpected cast operation.");
    }

    [[nodiscard]] Value *_translate_expression(XIRBuilder &b, const Expression *expr, bool load_lval) noexcept {
        LUISA_ASSERT(expr != nullptr, "Expression must not be null.");
        switch (expr->tag()) {
            case Expression::Tag::UNARY: return _translate_unary_expr(b, static_cast<const UnaryExpr *>(expr));
            case Expression::Tag::BINARY: return _translate_binary_expr(b, static_cast<const BinaryExpr *>(expr));
            case Expression::Tag::MEMBER: return _translate_member_expr(b, static_cast<const MemberExpr *>(expr), load_lval);
            case Expression::Tag::ACCESS: return _translate_member_or_access_expr(b, expr, load_lval);
            case Expression::Tag::LITERAL: return _translate_literal_expr(static_cast<const LiteralExpr *>(expr));
            case Expression::Tag::REF: return _translate_ref_expr(b, static_cast<const RefExpr *>(expr), load_lval);
            case Expression::Tag::CONSTANT: return _translate_constant_expr(static_cast<const ConstantExpr *>(expr));
            case Expression::Tag::CALL: return _translate_call_expr(b, static_cast<const CallExpr *>(expr));
            case Expression::Tag::CAST: return _translate_cast_expr(b, static_cast<const CastExpr *>(expr));
            case Expression::Tag::TYPE_ID: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::STRING_ID: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::FUNC_REF: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::CPUCUSTOM: LUISA_NOT_IMPLEMENTED();
            case Expression::Tag::GPUCUSTOM: LUISA_NOT_IMPLEMENTED();
        }
        return nullptr;
    }

    template<typename T>
    auto _commented(T inst) noexcept {
        for (auto comment : _current.comments) {
            inst->add_comment(comment->comment());
        }
        _current.comments.clear();
        return inst;
    }

    void _collect_comment(const Statement *stmt) noexcept {
        LUISA_ASSERT(stmt->tag() == Statement::Tag::COMMENT, "Unexpected statement type.");
        _current.comments.emplace_back(static_cast<const CommentStmt *>(stmt));
    }

    void _translate_switch_stmt(XIRBuilder &b, const SwitchStmt *ast_switch, luisa::span<const Statement *const> cdr) noexcept {
        // we do not support break/continue in switch statement
        auto old_break_continue_target = std::exchange(_current.break_continue_target, {});
        auto value = _translate_expression(b, ast_switch->expression(), true);
        auto inst = _commented(b.switch_(value));
        auto merge_block = inst->create_merge_block();
        auto case_break_removed = [](auto stmt_span) noexcept {
            while (!stmt_span.empty() &&
                   (stmt_span.back()->tag() == Statement::Tag::BREAK ||
                    stmt_span.back()->tag() == Statement::Tag::COMMENT)) {
                stmt_span = stmt_span.subspan(0, stmt_span.size() - 1);
            }
            return stmt_span;
        };
        for (auto s : ast_switch->body()->statements()) {
            switch (s->tag()) {
                case Statement::Tag::SWITCH_CASE: {
                    auto ast_case = static_cast<const SwitchCaseStmt *>(s);
                    LUISA_ASSERT(ast_case->expression()->tag() == Expression::Tag::LITERAL,
                                 "Unexpected switch case expression.");
                    auto ast_literal = static_cast<const LiteralExpr *>(ast_case->expression());
                    auto case_value = luisa::visit(
                        []<typename T>(T x) noexcept -> SwitchInst::case_value_type {
                            if constexpr (std::is_integral_v<T>) {
                                return static_cast<SwitchInst::case_value_type>(x);
                            } else {
                                LUISA_ERROR_WITH_LOCATION("Unexpected literal integer in switch case.");
                            }
                        },
                        ast_literal->value());
                    auto case_block = _commented(inst->create_case_block(case_value));
                    b.set_insertion_point(case_block);
                    auto case_stmts = case_break_removed(ast_case->body()->statements());
                    _translate_statements(b, case_stmts);
                    if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
                    break;
                }
                case Statement::Tag::SWITCH_DEFAULT: {
                    LUISA_ASSERT(inst->default_block() == nullptr,
                                 "Multiple default blocks in a switch statement.");
                    auto default_block = inst->create_default_block();
                    b.set_insertion_point(default_block);
                    auto ast_default = static_cast<const SwitchDefaultStmt *>(s);
                    auto case_stmts = case_break_removed(ast_default->body()->statements());
                    _translate_statements(b, case_stmts);
                    if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
                    break;
                }
                case Statement::Tag::COMMENT: {
                    _collect_comment(s);
                    break;
                }
                default: LUISA_ERROR_WITH_LOCATION("Unexpected statement in switch body.");
            }
        }
        if (inst->default_block() == nullptr) {
            b.set_insertion_point(inst->create_default_block());
            b.br(merge_block);
        }
        _current.break_continue_target = old_break_continue_target;
        b.set_insertion_point(merge_block);
        _translate_statements(b, cdr);
    }

    void _translate_if_stmt(XIRBuilder &b, const IfStmt *ast_if, luisa::span<const Statement *const> cdr) noexcept {
        auto cond = _translate_expression(b, ast_if->condition(), true);
        cond = b.static_cast_if_necessary(Type::of<bool>(), cond);
        auto inst = _commented(b.if_(cond));
        auto merge_block = inst->create_merge_block();
        // true branch
        {
            b.set_insertion_point(inst->create_true_block());
            _translate_statements(b, ast_if->true_branch()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
        }
        // false branch
        {
            b.set_insertion_point(inst->create_false_block());
            _translate_statements(b, ast_if->false_branch()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(merge_block); }
        }
        // merge block
        b.set_insertion_point(merge_block);
        _translate_statements(b, cdr);
    }

    void _translate_loop_stmt(XIRBuilder &b, const LoopStmt *ast_loop, luisa::span<const Statement *const> cdr) noexcept {
        auto inst = _commented(b.simple_loop());
        auto merge_block = inst->create_merge_block();
        auto body_block = inst->create_body_block();
        auto old_break_continue_target = std::exchange(_current.break_continue_target, {});
        // body block
        {
            _current.break_continue_target = {.break_target = merge_block,
                                              .continue_target = body_block};
            b.set_insertion_point(body_block);
            _translate_statements(b, ast_loop->body()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(inst->body_block()); }
            _current.break_continue_target = {};
        }
        // merge block
        _current.break_continue_target = old_break_continue_target;
        b.set_insertion_point(merge_block);
        _translate_statements(b, cdr);
    }

    void _translate_for_stmt(XIRBuilder &b, const ForStmt *ast_for, luisa::span<const Statement *const> cdr) noexcept {
        auto var = _translate_expression(b, ast_for->variable(), false);
        auto inst = _commented(b.loop());
        auto merge_block = inst->create_merge_block();
        auto prepare_block = inst->create_prepare_block();
        auto body_block = inst->create_body_block();
        auto update_block = inst->create_update_block();
        auto old_break_continue_target = std::exchange(_current.break_continue_target, {});
        // prepare block
        {
            b.set_insertion_point(prepare_block);
            auto cond = _translate_expression(b, ast_for->condition(), true);
            cond = b.static_cast_if_necessary(Type::of<bool>(), cond);
            b.cond_br(cond, body_block, merge_block);
        }
        // body block
        {
            _current.break_continue_target = {.break_target = merge_block,
                                              .continue_target = update_block};
            b.set_insertion_point(body_block);
            _translate_statements(b, ast_for->body()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(update_block); }
            _current.break_continue_target = {};
        }
        // update block
        {
            b.set_insertion_point(update_block);
            auto t = ast_for->variable()->type();
            // var += step
            auto step = _translate_expression(b, ast_for->step(), true);
            auto cast_step = _type_cast_if_necessary(b, t, step);
            auto prev = b.load(t, var);
            auto next = b.call(t, ArithmeticOp::BINARY_ADD, {prev, cast_step});
            b.store(var, next);
            // jump back to prepare block
            b.br(prepare_block);
        }
        // merge block
        _current.break_continue_target = old_break_continue_target;
        b.set_insertion_point(merge_block);
        _translate_statements(b, cdr);
    }

    void _translate_ray_query_stmt(XIRBuilder &b, const RayQueryStmt *ast_ray_query, luisa::span<const Statement *const> cdr) noexcept {
        // we do not support break/continue in ray query statement
        auto old_break_continue_target = std::exchange(_current.break_continue_target, {});
        // create the ray query loop
        auto loop_inst = _commented(b.ray_query_loop());
        auto dispatch_block = loop_inst->create_dispatch_block();
        auto merge_block = loop_inst->create_merge_block();
        // create the ray query dispatch block
        b.set_insertion_point(dispatch_block);
        auto query_object = _translate_expression(b, ast_ray_query->query(), false);
        auto dispatch_inst = _commented(b.ray_query_dispatch(query_object));
        dispatch_inst->set_exit_block(merge_block);
        // on surface candidate block
        {
            b.set_insertion_point(dispatch_inst->create_on_surface_candidate_block());
            _translate_statements(b, ast_ray_query->on_triangle_candidate()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(dispatch_block); }
        }
        // on procedural candidate block
        {
            b.set_insertion_point(dispatch_inst->create_on_procedural_candidate_block());
            _translate_statements(b, ast_ray_query->on_procedural_candidate()->statements());
            if (!b.is_insertion_point_terminator()) { b.br(dispatch_block); }
        }
        // merge block
        _current.break_continue_target = old_break_continue_target;
        b.set_insertion_point(merge_block);
        _translate_statements(b, cdr);
    }

    void _translate_statements(XIRBuilder &b, luisa::span<const Statement *const> stmts) noexcept {
        while (!stmts.empty()) {
            auto car = stmts.front();
            auto cdr = stmts.subspan(1);
            switch (car->tag()) {
                case Statement::Tag::BREAK: {
                    auto break_target = _current.break_continue_target.break_target;
                    LUISA_ASSERT(break_target != nullptr, "Invalid break statement.");
                    return static_cast<void>(_commented(b.break_(break_target)));
                }
                case Statement::Tag::CONTINUE: {
                    auto continue_target = _current.break_continue_target.continue_target;
                    LUISA_ASSERT(continue_target != nullptr, "Invalid continue statement.");
                    return static_cast<void>(_commented(b.continue_(continue_target)));
                }
                case Statement::Tag::RETURN: {
                    if (auto ast_expr = static_cast<const ReturnStmt *>(car)->expression()) {
                        auto value = _translate_expression(b, ast_expr, true);
                        return static_cast<void>(_commented(b.return_(value)));
                    }
                    return static_cast<void>(_commented(b.return_void()));
                }
                case Statement::Tag::SCOPE: LUISA_ERROR_WITH_LOCATION("Unexpected scope statement.");
                case Statement::Tag::IF: {
                    auto ast_if = static_cast<const IfStmt *>(car);
                    return _translate_if_stmt(b, ast_if, cdr);
                }
                case Statement::Tag::LOOP: {
                    auto ast_loop = static_cast<const LoopStmt *>(car);
                    return _translate_loop_stmt(b, ast_loop, cdr);
                }
                case Statement::Tag::EXPR: {
                    auto ast_expr = static_cast<const ExprStmt *>(car)->expression();
                    _commented(_translate_expression(b, ast_expr, false));
                    // in case the expression is a terminator, e.g., unreachable, we should stop here
                    if (b.insertion_point()->is_terminator()) { return; }
                    // otherwise, continue to the next statement
                    break;
                }
                case Statement::Tag::SWITCH: {
                    auto ast_switch = static_cast<const SwitchStmt *>(car);
                    return _translate_switch_stmt(b, ast_switch, cdr);
                }
                case Statement::Tag::SWITCH_CASE: LUISA_ERROR_WITH_LOCATION("Unexpected switch case statement.");
                case Statement::Tag::SWITCH_DEFAULT: LUISA_ERROR_WITH_LOCATION("Unexpected switch default statement.");
                case Statement::Tag::ASSIGN: {
                    auto assign = static_cast<const AssignStmt *>(car);
                    if (assign->lhs() != assign->rhs()) {
                        auto variable = _translate_expression(b, assign->lhs(), false);
                        auto value = _translate_expression(b, assign->rhs(), true);
                        value = _type_cast_if_necessary(b, variable->type(), value);
                        _commented(b.store(variable, value));
                    }
                    break;
                }
                case Statement::Tag::FOR: {
                    auto ast_for = static_cast<const ForStmt *>(car);
                    return _translate_for_stmt(b, ast_for, cdr);
                }
                case Statement::Tag::COMMENT: {
                    _collect_comment(car);
                    break;
                }
                case Statement::Tag::RAY_QUERY: {
                    auto ast_ray_query = static_cast<const RayQueryStmt *>(car);
                    return _translate_ray_query_stmt(b, ast_ray_query, cdr);
                }
                case Statement::Tag::AUTO_DIFF: LUISA_NOT_IMPLEMENTED();
                case Statement::Tag::PRINT: {
                    auto ast_print = static_cast<const PrintStmt *>(car);
                    luisa::fixed_vector<Value *, 16u> args;
                    for (auto ast_arg : ast_print->arguments()) {
                        args.emplace_back(_translate_expression(b, ast_arg, true));
                    }
                    _commented(b.print(luisa::string{ast_print->format()}, args));
                    break;
                }
            }
            // update the statement list
            stmts = cdr;
        }
    }

    void _translate_current_function() noexcept {
        // create the body block
        XIRBuilder b;
        b.set_insertion_point(_current.f->create_body_block());
        // convert the arguments
        for (auto ast_arg : _current.ast->arguments()) {
            auto arg = _current.f->create_argument(ast_arg.type(), ast_arg.is_reference());
            if (arg->is_value()) {
                // AST allows update of the argument, so we need to copy it to a local variable
                auto local = b.alloca_local(arg->type());
                local->add_comment("Local copy of argument");
                b.store(local, arg);
                _current.variables.emplace(ast_arg, local);
            } else {// otherwise, we can directly use the argument
                _current.variables.emplace(ast_arg, arg);
            }
        }
        for (auto ast_local : _current.ast->local_variables()) {
            LUISA_DEBUG_ASSERT(_current.variables.find(ast_local) == _current.variables.end(),
                               "Local variable already exists.");
            auto v = _current.variables.emplace(ast_local, b.alloca_local(ast_local.type())).first->second;
            if (ast_local.is_builtin()) {
                auto builtin_init = _translate_builtin_variable(ast_local);
                LUISA_ASSERT(v->type() == builtin_init->type(), "Variable type mismatch.");
                b.store(v, builtin_init);
            }
        }
        for (auto ast_shared : _current.ast->shared_variables()) {
            LUISA_DEBUG_ASSERT(_current.variables.find(ast_shared) == _current.variables.end(),
                               "Shared variable already exists.");
            _current.variables.emplace(ast_shared, b.alloca_shared(ast_shared.type()));
        }
        _translate_statements(b, _current.ast->body()->statements());
        if (!b.is_insertion_point_terminator()) {
            LUISA_ASSERT(_current.f->type() == nullptr,
                         "Non-void function must have a return statement at the end.");
            b.return_void();
        }
    }

public:
    explicit AST2XIRContext(const AST2XIRConfig &config) noexcept
        : _config{config}, _module{luisa::make_unique<Module>()} {}

    Function *add_function(const ASTFunction &f) noexcept {
        LUISA_ASSERT(_module != nullptr, "Module has been finalized.");
        // try emplace the function
        auto [iter, just_inserted] = _generated_functions.try_emplace(f.hash(), nullptr);
        // return the function if it has been translated
        if (!just_inserted) { return iter->second; }
        // create a new function
        auto def = [&]() noexcept -> FunctionDefinition * {
            switch (f.tag()) {
                case ASTFunction::Tag::KERNEL: {
                    auto kernel = _module->create_kernel();
                    kernel->set_block_size(f.block_size());
                    return kernel;
                }
                case ASTFunction::Tag::CALLABLE: {
                    return _module->create_callable(f.return_type());
                }
                case ASTFunction::Tag::RASTER_STAGE: LUISA_NOT_IMPLEMENTED();
            }
            LUISA_ERROR_WITH_LOCATION("Invalid function tag.");
        }();
        if (auto name = f.name(); !name.empty()) {
            def->set_name(name);
        }
        iter->second = def;
        // translate the function
        auto old = std::exchange(_current, {.f = def, .ast = &f});
        _translate_current_function();
        _current = old;
        return def;
    }

    Function *add_external_function(const ASTExternalFunction &f) noexcept {
        LUISA_ASSERT(_module != nullptr, "Module has been finalized.");
        LUISA_NOT_IMPLEMENTED();
    }

    [[nodiscard]] luisa::unique_ptr<Module> finalize() noexcept {
        return std::exchange(_module, nullptr);
    }
};

AST2XIRContext *ast_to_xir_translate_begin(const AST2XIRConfig &config) noexcept {
    return luisa::new_with_allocator<AST2XIRContext>(config);
}

void ast_to_xir_translate_add_function(AST2XIRContext *ctx, const ASTFunction &f) noexcept {
    ctx->add_function(f);
}

void ast_to_xir_translate_add_external_function(AST2XIRContext *ctx, const ASTExternalFunction &f) noexcept {
    ctx->add_external_function(f);
}

luisa::unique_ptr<Module> ast_to_xir_translate_finalize(AST2XIRContext *ctx) noexcept {
    auto m = ctx->finalize();
    luisa::delete_with_allocator(ctx);
    return m;
}

luisa::unique_ptr<Module> ast_to_xir_translate(const ASTFunction &kernel, const AST2XIRConfig &config) noexcept {
    AST2XIRContext ctx{config};
    ctx.add_function(kernel);
    return ctx.finalize();
}

}// namespace luisa::compute::xir
