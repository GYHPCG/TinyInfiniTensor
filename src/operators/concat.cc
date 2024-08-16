#include "operators/concat.h"
#include "utils/operator_utils.h"
namespace infini
{
    ConcatObj::ConcatObj(GraphObj *graph, TensorVec inputs, Tensor output, int _dim)
        : OperatorObj(OpType::Concat, inputs, {output})
    {
        int rank = inputs[0]->getRank();
        dim = get_real_axis(_dim, rank);
        IT_ASSERT(checkValid(graph));
    }

    optional<vector<Shape>> ConcatObj::inferShape(const TensorVec &inputs)
    {
        Shape dims = inputs[0]->getDims();
        auto rank = inputs[0]->getRank(); //

        // =================================== 作业 ===================================
        // TODO：修改 dims，返回正确的 concat 后的 shape
        // REF: https://onnx.ai/onnx/operators/onnx__Concat.html#concat-13
        // =================================== 作业 ===================================
        // 检查所有输入在非拼接轴上的维度是否相同
        for (const auto &input : inputs)
        {
            const auto &inputDims = input->getDims();
            for (size_t i = 0; i < rank; ++i)
            {
                if ((int)i != dim)
                {
                   IT_ASSERT(dims[i] == inputDims[i]);
                    // return   std::nullopt; // 如果非拼接轴的维度不一致，则返回 nullopt 表示形状推断失败
                }
                else
                {
                    // 累加拼接轴上的维度
                    dims[i] += inputDims[i];
                }
            }
        }
        return {{dims}};
    }

    std::string ConcatObj::toString() const
    {
        std::ostringstream os;
        os << "Concat[" << getGuid() << "]";
        os << "(";
        for (auto input : inputs)
            os << vecToString(input->getDims()) << ",";
        os << "dim=" << dim << ",";
        os << "input=";
        for (auto input : inputs)
            os << input->getGuid() << ",";
        os << "output=" << outputs[0]->getGuid() << ")";
        return os.str();
    }

} // namespace infini
