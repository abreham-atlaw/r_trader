#include "tensorflow/core/framework/op_kernel.h"


using namespace tensorflow;


class LinearKalmanFilterOp: OpKernel{

    public:
        explicit LinearKalmanFilterOp(OpKernelConstruction* context) : OpKernel(context){

        }

        void Compute(OpKernelContext* context) override {

            const Tensor& x_tensor = context->input(0);
            const Tensor& w_tensor = context->input(1);


            Tensor* y_tensor = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_output(0, x_tensor.shape(), &y_tensor)
            );

            Tensor* pvd_tensor = NULL;
            OP_REQUIRES_OK(
                context,
                context->allocate_temp(DataType::DT_FLOAT, TensorShape({3}), pvd_tensor)
            );

            auto x = x_tensor.flat<float>();
            auto w = w_tensor.flat<float>();
            auto y = y_tensor->flat<float>();
            auto pvd = pvd_tensor->flat<float>();

            pvd(0) = x(0);

            for(int i=0; i<x_tensor.shape().dim_size(0); i++){

                pvd(2) = x(0) - pvd(0);
                y(i) = (pvd(0) + (w(0) * pvd(2))) * w(2);
                pvd(1) += w(1) * pvd(2);
                pvd(0) = y(i) + pvd(1);

            }

        }
    

};