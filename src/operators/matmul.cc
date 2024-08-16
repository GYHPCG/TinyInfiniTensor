#include "operators/matmul.h"
#include "utils/operator_utils.h"
namespace infini
{

    MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA,
                         bool transB)
        : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}),
          transA(transA), transB(transB)
    {
        IT_ASSERT(checkValid(graph));
    }

    string MatmulObj::toString() const
    {
        std::ostringstream os;
        os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]")
           << ",A=" << inputs[0]->getGuid()
           << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid()
           << ",mnk=[" << m << "," << n << "," << k << "])";
        return os.str();
    }

    optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs)
    {
        // =================================== 作业 ===================================
        // TODO：返回经过 matmul 操作后的 shape
        // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
        // =================================== 作业 ===================================
        auto A_shape = inputs[0]->getDims();
        auto B_shape = inputs[1]->getDims();
        auto rank = A_shape.size();
        if (transA)
        {
            std::swap(A_shape[rank - 1], A_shape[rank - 2]);
        }
        if (transB)
        {
            std::swap(B_shape[rank - 1], B_shape[rank - 2]);
        }
        IT_ASSERT(A_shape[rank - 1] == B_shape[rank - 2]);
        auto m = A_shape[rank - 2];
        auto n = B_shape[rank - 1];

        A_shape[rank - 1] = n;
        B_shape[rank - 2] = m;

        auto result = infer_broadcast(A_shape, B_shape);
        return {{result}};
    }

} // namespace infini