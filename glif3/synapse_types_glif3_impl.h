/*! \file
 * \brief GLIF3 synapse type implementation with 4 independent alpha (double-exponential) synapses
 *
 * \details This implements 4 independent alpha synapses for the GLIF3 model,
 * matching the Chen et al. (2022) paper and TensorFlow implementation.
 *
 * Alpha synapse dynamics (Equation 3 from Chen et al.):
 *   C_rise(t+dt) = exp(-dt/tau) * C_rise(t) + (e/tau) * weight * spike
 *   I_syn(t+dt) = exp(-dt/tau) * I_syn(t) + dt * exp(-dt/tau) * C_rise(t)
 */

#ifndef _SYNAPSE_TYPES_GLIF3_IMPL_H_
#define _SYNAPSE_TYPES_GLIF3_IMPL_H_

#include <neuron/synapse_types/exp_synapse_utils.h>
#include <debug.h>

#define SYNAPSE_TYPE_BITS 2
#define SYNAPSE_TYPE_COUNT 4

struct synapse_types_params_t {
    exp_params_t syn_0;
    exp_params_t syn_1;
    exp_params_t syn_2;
    exp_params_t syn_3;
    REAL time_step_ms;
};

struct synapse_types_t {
    exp_state_t syn_0_rise;
    exp_state_t syn_0;
    exp_state_t syn_1_rise;
    exp_state_t syn_1;
    exp_state_t syn_2_rise;
    exp_state_t syn_2;
    exp_state_t syn_3_rise;
    exp_state_t syn_3;
};

#define NUM_EXCITATORY_RECEPTORS 4
#define NUM_INHIBITORY_RECEPTORS 0

#include <neuron/synapse_types/synapse_types.h>

typedef enum input_buffer_regions {
    SYNAPSE_0, SYNAPSE_1, SYNAPSE_2, SYNAPSE_3
} input_buffer_regions;

#define INIT_SYNAPSE_PAIR(state, params, syn_num) \
    decay_and_init(&state->syn_##syn_num##_rise, &params->syn_##syn_num, params->time_step_ms, n_steps_per_timestep); \
    decay_and_init(&state->syn_##syn_num, &params->syn_##syn_num, params->time_step_ms, n_steps_per_timestep); \
    state->syn_##syn_num.synaptic_input_value = params->syn_##syn_num.init_input

static inline void synapse_types_initialise(synapse_types_t *state,
        synapse_types_params_t *params, uint32_t n_steps_per_timestep) {
    INIT_SYNAPSE_PAIR(state, params, 0);
    INIT_SYNAPSE_PAIR(state, params, 1);
    INIT_SYNAPSE_PAIR(state, params, 2);
    INIT_SYNAPSE_PAIR(state, params, 3);
}

#define SAVE_SYNAPSE(state, params, syn_num) \
    params->syn_##syn_num.init_input = state->syn_##syn_num.synaptic_input_value

static inline void synapse_types_save_state(synapse_types_t *state,
        synapse_types_params_t *params) {
    SAVE_SYNAPSE(state, params, 0);
    SAVE_SYNAPSE(state, params, 1);
    SAVE_SYNAPSE(state, params, 2);
    SAVE_SYNAPSE(state, params, 3);
}

// Alpha synapse: I_syn = decay*I_syn + decay*C_rise (dt=1ms implicit)
#define SHAPE_ALPHA_SYNAPSE(p, n) do { \
    exp_shaping(&p->syn_##n##_rise); \
    p->syn_##n.synaptic_input_value = \
        decay_s1615(p->syn_##n.synaptic_input_value, p->syn_##n.decay) + \
        decay_s1615(p->syn_##n##_rise.synaptic_input_value, p->syn_##n.decay); \
} while(0)

static inline void synapse_types_shape_input(synapse_types_t *parameters) {
    SHAPE_ALPHA_SYNAPSE(parameters, 0);
    SHAPE_ALPHA_SYNAPSE(parameters, 1);
    SHAPE_ALPHA_SYNAPSE(parameters, 2);
    SHAPE_ALPHA_SYNAPSE(parameters, 3);
}

static inline void synapse_types_add_neuron_input(
        index_t synapse_type_index, synapse_types_t *parameters,
        input_t input) {
    switch (synapse_type_index) {
        case SYNAPSE_0: add_input_exp(&parameters->syn_0_rise, input); break;
        case SYNAPSE_1: add_input_exp(&parameters->syn_1_rise, input); break;
        case SYNAPSE_2: add_input_exp(&parameters->syn_2_rise, input); break;
        case SYNAPSE_3: add_input_exp(&parameters->syn_3_rise, input); break;
    }
}

static inline input_t* synapse_types_get_excitatory_input(
        input_t *excitatory_response, synapse_types_t *parameters) {
    excitatory_response[0] = parameters->syn_0.synaptic_input_value;
    excitatory_response[1] = parameters->syn_1.synaptic_input_value;
    excitatory_response[2] = parameters->syn_2.synaptic_input_value;
    excitatory_response[3] = parameters->syn_3.synaptic_input_value;
    return &excitatory_response[0];
}

static inline input_t* synapse_types_get_inhibitory_input(
        input_t *inhibitory_response, synapse_types_t *parameters) {
    use(inhibitory_response);
    use(parameters);
    return NULL;
}

static inline const char *synapse_types_get_type_char(index_t synapse_type_index) {
    return (synapse_type_index < 4) ? "0" : "?";
}

static inline void synapse_types_print_input(synapse_types_t *parameters) {
    use(parameters);
}

static inline void synapse_types_print_parameters(synapse_types_t *parameters) {
    use(parameters);
}

#endif  // _SYNAPSE_TYPES_GLIF3_IMPL_H_
