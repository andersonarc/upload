/*! \file
 * \brief GLIF3 alpha synapse implementation (memory-optimized)
 *
 * Alpha synapse dynamics (Chen et al. 2022, Eq. 3):
 *   C_rise(t+dt) = exp(-dt/tau) * C_rise(t) + (e/tau) * weight * spike
 *   I_syn(t+dt) = exp(-dt/tau) * I_syn(t) + exp(-dt/tau) * C_rise(t)
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

static inline void synapse_types_initialise(synapse_types_t *state,
        synapse_types_params_t *params, uint32_t n_steps_per_timestep) {
    decay_and_init(&state->syn_0_rise, &params->syn_0, params->time_step_ms, n_steps_per_timestep);
    decay_and_init(&state->syn_0, &params->syn_0, params->time_step_ms, n_steps_per_timestep);
    state->syn_0.synaptic_input_value = params->syn_0.init_input;

    decay_and_init(&state->syn_1_rise, &params->syn_1, params->time_step_ms, n_steps_per_timestep);
    decay_and_init(&state->syn_1, &params->syn_1, params->time_step_ms, n_steps_per_timestep);
    state->syn_1.synaptic_input_value = params->syn_1.init_input;

    decay_and_init(&state->syn_2_rise, &params->syn_2, params->time_step_ms, n_steps_per_timestep);
    decay_and_init(&state->syn_2, &params->syn_2, params->time_step_ms, n_steps_per_timestep);
    state->syn_2.synaptic_input_value = params->syn_2.init_input;

    decay_and_init(&state->syn_3_rise, &params->syn_3, params->time_step_ms, n_steps_per_timestep);
    decay_and_init(&state->syn_3, &params->syn_3, params->time_step_ms, n_steps_per_timestep);
    state->syn_3.synaptic_input_value = params->syn_3.init_input;
}

static inline void synapse_types_save_state(synapse_types_t *state,
        synapse_types_params_t *params) {
    params->syn_0.init_input = state->syn_0.synaptic_input_value;
    params->syn_1.init_input = state->syn_1.synaptic_input_value;
    params->syn_2.init_input = state->syn_2.synaptic_input_value;
    params->syn_3.init_input = state->syn_3.synaptic_input_value;
}

static inline void synapse_types_shape_input(synapse_types_t *p) {
    // Alpha synapse dynamics (Chen et al. 2022, Eq. 3)
    exp_shaping(&p->syn_0_rise);
    p->syn_0.synaptic_input_value =
        decay_s1615(p->syn_0.synaptic_input_value, p->syn_0.decay) +
        decay_s1615(p->syn_0_rise.synaptic_input_value, p->syn_0.decay);

    exp_shaping(&p->syn_1_rise);
    p->syn_1.synaptic_input_value =
        decay_s1615(p->syn_1.synaptic_input_value, p->syn_1.decay) +
        decay_s1615(p->syn_1_rise.synaptic_input_value, p->syn_1.decay);

    exp_shaping(&p->syn_2_rise);
    p->syn_2.synaptic_input_value =
        decay_s1615(p->syn_2.synaptic_input_value, p->syn_2.decay) +
        decay_s1615(p->syn_2_rise.synaptic_input_value, p->syn_2.decay);

    exp_shaping(&p->syn_3_rise);
    p->syn_3.synaptic_input_value =
        decay_s1615(p->syn_3.synaptic_input_value, p->syn_3.decay) +
        decay_s1615(p->syn_3_rise.synaptic_input_value, p->syn_3.decay);
}

static inline void synapse_types_add_neuron_input(
        index_t synapse_type_index, synapse_types_t *parameters,
        input_t input) {
    // Add input to rise variable (alpha synapse first stage)
    switch (synapse_type_index) {
        case SYNAPSE_0:
            add_input_exp(&parameters->syn_0_rise, input);
            break;
        case SYNAPSE_1:
            add_input_exp(&parameters->syn_1_rise, input);
            break;
        case SYNAPSE_2:
            add_input_exp(&parameters->syn_2_rise, input);
            break;
        case SYNAPSE_3:
            add_input_exp(&parameters->syn_3_rise, input);
            break;
    }
}

static inline input_t* synapse_types_get_excitatory_input(
        input_t *excitatory_response, synapse_types_t *parameters) {
    // Return current values (I_syn), not rise values
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
    use(synapse_type_index);
    return "0";
}

static inline void synapse_types_print_input(synapse_types_t *parameters) {
    use(parameters);
}

static inline void synapse_types_print_parameters(synapse_types_t *parameters) {
    use(parameters);
}

#endif  // _SYNAPSE_TYPES_GLIF3_IMPL_H_
