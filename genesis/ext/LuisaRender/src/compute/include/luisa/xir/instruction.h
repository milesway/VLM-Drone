#pragma once

#include <luisa/core/concepts.h>
#include <luisa/xir/user.h>

namespace luisa::compute::xir {

class BasicBlock;
class Function;

enum struct DerivedInstructionTag {

    /* control flow instructions */
    IF,                // basic block terminator: conditional branches
    SWITCH,            // basic block terminator: switch branches
    LOOP,              // basic block terminator: loops
    SIMPLE_LOOP,       // basic block terminator: simple (do-while) loops
    BRANCH,            // basic block terminator: unconditional branches
    CONDITIONAL_BRANCH,// basic block terminator: conditional branches
    UNREACHABLE,       // basic block terminator: unreachable
    BREAK,             // basic block terminator: break (removed after control flow normalization)
    CONTINUE,          // basic block terminator: continue (removed after control flow normalization)
    RETURN,            // basic block terminator: return (early returns are removed after control flow normalization)
    RASTER_DISCARD,    // basic block terminator: raster discard

    /* PHI nodes */
    PHI,// basic block beginning: phi nodes

    /* variable instructions */
    ALLOCA,
    LOAD,
    STORE,
    GEP,

    /* atomic instructions */
    ATOMIC,// operates on buffers or shared memory

    /* ALU (arithmetic logic unit) instructions */
    ARITHMETIC,// arithmetic operations

    /* thread-group instructions */
    THREAD_GROUP,// volatile, may involve synchronization and cannot be moved/eliminated

    /* resource instructions */
    RESOURCE_QUERY,// query resource properties, free to move and eliminate
    RESOURCE_READ, // read from resources, may be eliminated if not used, but can be volatile to code motion
    RESOURCE_WRITE,// write to resources, may be volatile to code elimination and motion

    /* ray query instructions */
    RAY_QUERY_LOOP,        // basic block beginning: ray query loop
    RAY_QUERY_DISPATCH,    // basic block terminator: ray query switch branches
    RAY_QUERY_OBJECT_READ, // read from ray query objects
    RAY_QUERY_OBJECT_WRITE,// write to ray query objects
    RAY_QUERY_PIPELINE,    // ray query pipeline with surface and procedural callbacks

    /* automatic differentiation */
    AUTODIFF_SCOPE,
    AUTODIFF_INTRINSIC,

    /* other instructions */
    CALL, // user or external function calls
    CAST, // type casts
    PRINT,// kernel print
    CLOCK,// kernel clock

    ASSERT,// assertion
    ASSUME,// assumption

    OUTLINE,// mark that the body might be outlined (e.g., for faster compilation)
};

[[nodiscard]] constexpr luisa::string_view to_string(DerivedInstructionTag tag) noexcept {
    using namespace std::string_view_literals;
    switch (tag) {
        case DerivedInstructionTag::IF: return "if"sv;
        case DerivedInstructionTag::SWITCH: return "switch"sv;
        case DerivedInstructionTag::LOOP: return "loop"sv;
        case DerivedInstructionTag::SIMPLE_LOOP: return "simple_loop"sv;
        case DerivedInstructionTag::BRANCH: return "branch"sv;
        case DerivedInstructionTag::CONDITIONAL_BRANCH: return "conditional_branch"sv;
        case DerivedInstructionTag::UNREACHABLE: return "unreachable"sv;
        case DerivedInstructionTag::BREAK: return "break"sv;
        case DerivedInstructionTag::CONTINUE: return "continue"sv;
        case DerivedInstructionTag::RETURN: return "return"sv;
        case DerivedInstructionTag::RASTER_DISCARD: return "raster_discard"sv;
        case DerivedInstructionTag::PHI: return "phi"sv;
        case DerivedInstructionTag::ALLOCA: return "alloca"sv;
        case DerivedInstructionTag::LOAD: return "load"sv;
        case DerivedInstructionTag::STORE: return "store"sv;
        case DerivedInstructionTag::GEP: return "gep"sv;
        case DerivedInstructionTag::ATOMIC: return "atomic"sv;
        case DerivedInstructionTag::ARITHMETIC: return "arithmetic"sv;
        case DerivedInstructionTag::THREAD_GROUP: return "thread_group"sv;
        case DerivedInstructionTag::RESOURCE_QUERY: return "resource_query"sv;
        case DerivedInstructionTag::RESOURCE_READ: return "resource_read"sv;
        case DerivedInstructionTag::RESOURCE_WRITE: return "resource_write"sv;
        case DerivedInstructionTag::RAY_QUERY_LOOP: return "ray_query_loop"sv;
        case DerivedInstructionTag::RAY_QUERY_DISPATCH: return "ray_query_dispatch"sv;
        case DerivedInstructionTag::RAY_QUERY_OBJECT_READ: return "ray_query_object_read"sv;
        case DerivedInstructionTag::RAY_QUERY_OBJECT_WRITE: return "ray_query_object_write"sv;
        case DerivedInstructionTag::RAY_QUERY_PIPELINE: return "ray_query_pipeline"sv;
        case DerivedInstructionTag::CALL: return "call"sv;
        case DerivedInstructionTag::CAST: return "cast"sv;
        case DerivedInstructionTag::PRINT: return "print"sv;
        case DerivedInstructionTag::CLOCK: return "clock"sv;
        case DerivedInstructionTag::ASSERT: return "assert"sv;
        case DerivedInstructionTag::ASSUME: return "assume"sv;
        case DerivedInstructionTag::OUTLINE: return "outline"sv;
        case DerivedInstructionTag::AUTODIFF_SCOPE: return "autodiff_scope"sv;
        case DerivedInstructionTag::AUTODIFF_INTRINSIC: return "autodiff_intrinsic"sv;
    }
    return "unknown"sv;
}

class ControlFlowMerge;

struct InstructionCloneValueResolver {
    virtual ~InstructionCloneValueResolver() noexcept = default;
    [[nodiscard]] virtual Value *resolve(const Value *value) noexcept = 0;
};

class XIRBuilder;

class LC_XIR_API Instruction : public IntrusiveNode<Instruction, DerivedBlockScopeValue<Instruction, DerivedValueTag::INSTRUCTION, User>> {

protected:
    void _remove_self_from_operand_use_lists() noexcept;
    void _add_self_to_operand_use_lists() noexcept;
    [[nodiscard]] bool _should_add_self_to_operand_use_lists() const noexcept override;

public:
    explicit Instruction(BasicBlock *parent_block, const Type *type) noexcept;
    [[nodiscard]] virtual DerivedInstructionTag derived_instruction_tag() const noexcept = 0;
    [[nodiscard]] virtual Instruction *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept = 0;

    void remove_self() noexcept override;
    void insert_before_self(Instruction *node) noexcept override;
    void insert_after_self(Instruction *node) noexcept override;
    void replace_self_with(Instruction *node) noexcept;

    [[nodiscard]] virtual bool is_terminator() const noexcept { return false; }

    [[nodiscard]] virtual ControlFlowMerge *control_flow_merge() noexcept { return nullptr; }
    [[nodiscard]] const ControlFlowMerge *control_flow_merge() const noexcept;

    LUISA_XIR_DEFINED_ISA_METHOD(Instruction, instruction)
};

class LC_XIR_API SentinelInst final : public Instruction {
public:
    explicit SentinelInst(BasicBlock *parent_block) noexcept;
    [[nodiscard]] DerivedInstructionTag derived_instruction_tag() const noexcept override;
    [[nodiscard]] Instruction *clone(XIRBuilder &b, InstructionCloneValueResolver &resolver) const noexcept override;
};

using InstructionList = InlineIntrusiveList<Instruction, SentinelInst>;

class LC_XIR_API TerminatorInstruction : public Instruction {
public:
    explicit TerminatorInstruction(BasicBlock *block) noexcept;
    [[nodiscard]] bool is_terminator() const noexcept final { return true; }
};

// unconditional branch
class LC_XIR_API BranchTerminatorInstruction : public TerminatorInstruction {

public:
    static constexpr size_t operand_index_target = 0u;
    static constexpr size_t derived_operand_index_offset = 1u;

public:
    explicit BranchTerminatorInstruction(BasicBlock *parent_block) noexcept;

    void set_target_block(BasicBlock *target) noexcept;
    BasicBlock *create_target_block(bool overwrite_existing = false) noexcept;

    [[nodiscard]] BasicBlock *target_block() noexcept;
    [[nodiscard]] const BasicBlock *target_block() const noexcept;
};

// conditional branch
class LC_XIR_API ConditionalBranchTerminatorInstruction : public TerminatorInstruction {

public:
    static constexpr size_t operand_index_condition = 0u;
    static constexpr size_t operand_index_true_target = 1u;
    static constexpr size_t operand_index_false_target = 2u;
    static constexpr size_t derived_operand_index_offset = 3u;

public:
    explicit ConditionalBranchTerminatorInstruction(BasicBlock *parent_block,
                                                    Value *condition = nullptr) noexcept;

    void set_condition(Value *condition) noexcept;
    void set_true_target(BasicBlock *target) noexcept;
    void set_false_target(BasicBlock *target) noexcept;

    BasicBlock *create_true_block(bool overwrite_existing = false) noexcept;
    BasicBlock *create_false_block(bool overwrite_existing = false) noexcept;

    [[nodiscard]] Value *condition() noexcept;
    [[nodiscard]] const Value *condition() const noexcept;

    [[nodiscard]] BasicBlock *true_block() noexcept;
    [[nodiscard]] const BasicBlock *true_block() const noexcept;

    [[nodiscard]] BasicBlock *false_block() noexcept;
    [[nodiscard]] const BasicBlock *false_block() const noexcept;
};

template<typename Derived, DerivedInstructionTag tag, typename Base = Instruction>
    requires std::derived_from<Base, Instruction>
class DerivedInstruction : public Base {
public:
    using derived_instruction_type = Derived;
    using Super = DerivedInstruction;
    using Base::Base;

    [[nodiscard]] static constexpr DerivedInstructionTag
    static_derived_instruction_tag() noexcept { return tag; }

    [[nodiscard]] DerivedInstructionTag
    derived_instruction_tag() const noexcept final {
        return static_derived_instruction_tag();
    }
};

template<typename Derived, DerivedInstructionTag tag>
class DerivedTerminatorInstruction : public DerivedInstruction<Derived, tag, TerminatorInstruction> {
public:
    using Super = DerivedTerminatorInstruction;
    using DerivedInstruction<Derived, tag, TerminatorInstruction>::DerivedInstruction;
};

template<typename Derived, DerivedInstructionTag tag>
class DerivedBranchInstruction : public DerivedInstruction<Derived, tag, BranchTerminatorInstruction> {
public:
    using Super = DerivedBranchInstruction;
    using DerivedInstruction<Derived, tag, BranchTerminatorInstruction>::DerivedInstruction;
};

template<typename Derived, DerivedInstructionTag tag>
class DerivedConditionalBranchInstruction : public DerivedInstruction<Derived, tag, ConditionalBranchTerminatorInstruction> {
public:
    using Super = DerivedConditionalBranchInstruction;
    using DerivedInstruction<Derived, tag, ConditionalBranchTerminatorInstruction>::DerivedInstruction;
};

class LC_XIR_API ControlFlowMerge : luisa::concepts::Noncopyable {

private:
    BasicBlock *_merge_block{nullptr};

protected:
    ControlFlowMerge() noexcept = default;
    ~ControlFlowMerge() noexcept = default;

private:
    [[nodiscard]] virtual Instruction *_base_instruction() noexcept = 0;

public:
    void set_merge_block(BasicBlock *block) noexcept;
    [[nodiscard]] BasicBlock *merge_block() noexcept { return _merge_block; }
    [[nodiscard]] const BasicBlock *merge_block() const noexcept { return _merge_block; }
    BasicBlock *create_merge_block(bool overwrite_existing = false) noexcept;
};

template<typename Base>
    requires std::derived_from<Base, Instruction>
class ControlFlowMergeMixin : public Base,
                              public ControlFlowMerge {
private:
    [[nodiscard]] Instruction *_base_instruction() noexcept final {
        return static_cast<Instruction *>(this);
    }

public:
    using Super = ControlFlowMergeMixin;
    using Base::Base;
    [[nodiscard]] ControlFlowMerge *control_flow_merge() noexcept final { return this; }
};

template<typename OpType>
class InstructionOpMixin {

private:
    OpType _op;

public:
    explicit InstructionOpMixin(OpType op) noexcept : _op{op} {}
    [[nodiscard]] OpType op() const noexcept { return _op; }
    void set_op(OpType op) noexcept { _op = op; }
};

}// namespace luisa::compute::xir
