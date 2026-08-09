#include "vector_env.h"
#include <stdexcept>

VectorEnv::VectorEnv(int count, int width, int height, unsigned int base_seed, int step_limit)
{
    if (count < 1)
    {
        throw std::invalid_argument("VectorEnv needs at least one game");
    }
    envs_.reserve(count);
    for (int index = 0; index < count; index++)
    {
        // One generator per game, offset from the base, so a run reproduces and
        // no two games share a food sequence.
        envs_.emplace_back(width, height, base_seed + index, step_limit);
    }
}

void VectorEnv::resetAll()
{
    for (SnakeEnv& env : envs_)
    {
        env.reset();
    }
}

void VectorEnv::resetOne(int index)
{
    envs_[index].reset();
}

void VectorEnv::step(const SnakeEnv::Action* actions, SnakeEnv::StepResult* results_out)
{
    for (int index = 0; index < count(); index++)
    {
        SnakeEnv& env = envs_[index];
        if (env.done())
        {
            // Report the terminal state rather than stepping it. SnakeEnv::step
            // throws on a finished episode, and it should - a trainer that has
            // lost track of which games are live has a bug worth surfacing.
            results_out[index] = { 0.0f, true, env.won() };
            continue;
        }
        results_out[index] = env.step(actions[index]);
    }
}

void VectorEnv::encodeAll(float* batch_out) const
{
    const int stride = encodedSizePerEnv();
    for (int index = 0; index < count(); index++)
    {
        envs_[index].encode(batch_out + static_cast<size_t>(index) * stride);
    }
}
