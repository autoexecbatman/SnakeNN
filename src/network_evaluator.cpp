#include <torch/torch.h>

#include <stdexcept>

#include "az_parameters.h"
#include "network_evaluator.h"

NetworkEvaluator::NetworkEvaluator(AlphaZeroNet network, torch::Device device)
    : network_(network), device_(device), evaluations_(0)
{
    network_->to(device_);
    network_->eval();
}

void NetworkEvaluator::evaluate(const std::vector<const SnakeEnv*>& states, float* priors_out,
                                float* values_out, float* steps_out, float* death_risk_out)
{
    if (states.empty())
    {
        return;
    }

    const int batch = static_cast<int>(states.size());
    const int per_state = states[0]->encodedSize();
    const int width = states[0]->width();
    const int height = states[0]->height();

    staging_.resize(static_cast<size_t>(batch) * per_state);
    for (int index = 0; index < batch; index++)
    {
        if (states[index]->encodedSize() != per_state)
        {
            throw std::invalid_argument("evaluator received mixed board sizes in one batch");
        }
        states[index]->encode(staging_.data() + static_cast<size_t>(index) * per_state);
    }

    torch::NoGradGuard no_grad;

    // from_blob borrows the staging buffer rather than copying it; the copy to
    // the device happens once, in .to().
    torch::Tensor input =
        torch::from_blob(staging_.data(), { batch, SnakeEnv::PLANE_COUNT, height, width },
                         torch::TensorOptions().dtype(torch::kFloat32))
            .to(device_);

    const Prediction prediction = network_->forward(input);
    torch::Tensor priors = torch::softmax(prediction.policy_logits, 1).to(torch::kCPU).contiguous();
    torch::Tensor values = prediction.value.to(torch::kCPU).contiguous();
    torch::Tensor steps = prediction.steps_to_go.to(torch::kCPU).contiguous();
    torch::Tensor death_risk = prediction.death_risk.to(torch::kCPU).contiguous();

    const float* prior_data = priors.data_ptr<float>();
    const float* value_data = values.data_ptr<float>();
    const float* steps_data = steps.data_ptr<float>();
    const float* death_data = death_risk.data_ptr<float>();
    for (int index = 0; index < batch * SnakeEnv::ACTION_COUNT; index++)
    {
        priors_out[index] = prior_data[index];
        // Zero until the head has a loss behind it. A head that no target has
        // trained emits its initialisation, which at a leaf is noise the cap
        // would act on; zero leaves the risk to the simulator's own terminals,
        // which is what the first measurement is meant to be of.
        death_risk_out[index] = az::DEATH_RISK_FROM_NETWORK ? death_data[index] : 0.0f;
    }
    for (int index = 0; index < batch; index++)
    {
        values_out[index] = value_data[index];
        steps_out[index] = steps_data[index];
    }

    evaluations_ += batch;
}
