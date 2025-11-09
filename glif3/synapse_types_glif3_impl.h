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

// 4 synapse types require 2 bits (2^2 = 4)
#define SYNAPSE_TYPE_BITS 2

// 4 synapse types for GLIF3 (tau_syn0, tau_syn1, tau_syn2, tau_syn3)
#define SYNAPSE_TYPE_COUNT 4

// Parameters for 4 independent alpha synapses
struct synapse_types_params_t {
    exp_params_t syn_0;  // tau_syn0
    exp_params_t syn_1;  // tau_syn1
    exp_params_t syn_2;  // tau_syn2
    exp_params_t syn_3;  // tau_syn3
    REAL time_step_ms;
};

// State for 4 independent alpha synapses
// Each alpha synapse requires TWO state variables: rise (C_rise) and current (I_syn)
struct synapse_types_t {
    exp_state_t syn_0_rise;  // C_rise for synapse 0
    exp_state_t syn_0;       // I_syn for synapse 0
    exp_state_t syn_1_rise;  // C_rise for synapse 1
    exp_state_t syn_1;       // I_syn for synapse 1
    exp_state_t syn_2_rise;  // C_rise for synapse 2
    exp_state_t syn_2;       // I_syn for synapse 2
    exp_state_t syn_3_rise;  // C_rise for synapse 3
    exp_state_t syn_3;       // I_syn for synapse 3
};

// All 4 receptors are excitatory (can be made inhibitory by connection weight sign)
#define NUM_EXCITATORY_RECEPTORS 4
#define NUM_INHIBITORY_RECEPTORS 0

// Include this after defining the above
#include <neuron/synapse_types/synapse_types.h>

// Synapse type indices
typedef enum input_buffer_regions {
    SYNAPSE_0, SYNAPSE_1, SYNAPSE_2, SYNAPSE_3
} input_buffer_regions;

static inline void synapse_types_initialise(synapse_types_t *state,
        synapse_types_params_t *params, uint32_t n_steps_per_timestep) {
    // Initialize both rise and current states for each synapse
    // Both use the same tau from params
    decay_and_init(&state->syn_0_rise, &params->syn_0, params->time_step_ms, n_steps_per_timestep);
    decay_and_init(&state->syn_0, &params->syn_0, params->time_step_ms, n_steps_per_timestep);

    decay_and_init(&state->syn_1_rise, &params->syn_1, params->time_step_ms, n_steps_per_timestep);
    decay_and_init(&state->syn_1, &params->syn_1, params->time_step_ms, n_steps_per_timestep);

    decay_and_init(&state->syn_2_rise, &params->syn_2, params->time_step_ms, n_steps_per_timestep);
    decay_and_init(&state->syn_2, &params->syn_2, params->time_step_ms, n_steps_per_timestep);

    decay_and_init(&state->syn_3_rise, &params->syn_3, params->time_step_ms, n_steps_per_timestep);
    decay_and_init(&state->syn_3, &params->syn_3, params->time_step_ms, n_steps_per_timestep);

    // Set initial current values to zero (rise variables already initialized)
    state->syn_0.synaptic_input_value = params->syn_0.init_input;
    state->syn_1.synaptic_input_value = params->syn_1.init_input;
    state->syn_2.synaptic_input_value = params->syn_2.init_input;
    state->syn_3.synaptic_input_value = params->syn_3.init_input;
}

static inline void synapse_types_save_state(synapse_types_t *state,
        synapse_types_params_t *params) {
    // Save the current (I_syn) values for continuation
    params->syn_0.init_input = state->syn_0.synaptic_input_value;
    params->syn_1.init_input = state->syn_1.synaptic_input_value;
    params->syn_2.init_input = state->syn_2.synaptic_input_value;
    params->syn_3.init_input = state->syn_3.synaptic_input_value;
}

//! \brief Shapes the synaptic input (alpha/double-exponential decay)
//! \details Implements Chen et al. Equation 3:
//!   C_rise(t+dt) = exp(-dt/tau) * C_rise(t)  [spike input added separately]
//!   I_syn(t+dt) = exp(-dt/tau) * I_syn(t) + dt * exp(-dt/tau) * C_rise(t)
static inline void synapse_types_shape_input(synapse_types_t *parameters) {
    REAL dt = 1.0k;  // Timestep in ms

    // Synapse 0
    exp_shaping(&parameters->syn_0_rise);
    parameters->syn_0.synaptic_input_value =
        decay_s1615(parameters->syn_0.synaptic_input_value, parameters->syn_0.decay) +
        dt * decay_s1615(parameters->syn_0_rise.synaptic_input_value, parameters->syn_0.decay);

    // Synapse 1
    exp_shaping(&parameters->syn_1_rise);
    parameters->syn_1.synaptic_input_value =
        decay_s1615(parameters->syn_1.synaptic_input_value, parameters->syn_1.decay) +
        dt * decay_s1615(parameters->syn_1_rise.synaptic_input_value, parameters->syn_1.decay);

    // Synapse 2
    exp_shaping(&parameters->syn_2_rise);
    parameters->syn_2.synaptic_input_value =
        decay_s1615(parameters->syn_2.synaptic_input_value, parameters->syn_2.decay) +
        dt * decay_s1615(parameters->syn_2_rise.synaptic_input_value, parameters->syn_2.decay);

    // Synapse 3
    exp_shaping(&parameters->syn_3_rise);
    parameters->syn_3.synaptic_input_value =
        decay_s1615(parameters->syn_3.synaptic_input_value, parameters->syn_3.decay) +
        dt * decay_s1615(parameters->syn_3_rise.synaptic_input_value, parameters->syn_3.decay);
}

//! \brief Adds input to the appropriate synapse type
//! \details For alpha synapses, input is added to the RISE variable (C_rise)
//!   C_rise += (e/tau) * weight * spike
//! The 'init' parameter already contains (e/tau) scaling
static inline void synapse_types_add_neuron_input(
        index_t synapse_type_index, synapse_types_t *parameters,
        input_t input) {
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
        default:
            log_error("Invalid synapse type index: %d", synapse_type_index);
            break;
    }
}

//! \brief Gets all 4 excitatory inputs
//! \details Returns the CURRENT (I_syn) values, not rise values
static inline input_t* synapse_types_get_excitatory_input(
        input_t *excitatory_response, synapse_types_t *parameters) {
    excitatory_response[0] = parameters->syn_0.synaptic_input_value;
    excitatory_response[1] = parameters->syn_1.synaptic_input_value;
    excitatory_response[2] = parameters->syn_2.synaptic_input_value;
    excitatory_response[3] = parameters->syn_3.synaptic_input_value;
    return &excitatory_response[0];
}

//! \brief No inhibitory inputs (returns NULL)
static inline input_t* synapse_types_get_inhibitory_input(
        input_t *inhibitory_response, synapse_types_t *parameters) {
    use(inhibitory_response);
    use(parameters);
    return NULL;
}

//! \brief Returns character for synapse type (for debug)
static inline const char *synapse_types_get_type_char(
        index_t synapse_type_index) {
    switch (synapse_type_index) {
        case SYNAPSE_0:
            return "0";
        case SYNAPSE_1:
            return "1";
        case SYNAPSE_2:
            return "2";
        case SYNAPSE_3:
            return "3";
        default:
            log_debug("Did not recognise synapse type %i", synapse_type_index);
            return "?";
    }
}

//! \brief Prints input for debug purposes
static inline void synapse_types_print_input(synapse_types_t *parameters) {
    io_printf(IO_BUF, "%12.6k - %12.6k - %12.6k - %12.6k",
            parameters->syn_0.synaptic_input_value,
            parameters->syn_1.synaptic_input_value,
            parameters->syn_2.synaptic_input_value,
            parameters->syn_3.synaptic_input_value);
}

//! \brief Prints parameters for debug purposes
static inline void synapse_types_print_parameters(synapse_types_t *parameters) {
    log_info("syn_0_rise_decay = %R\n", (unsigned fract) parameters->syn_0_rise.decay);
    log_info("syn_0_rise_init  = %R\n", (unsigned fract) parameters->syn_0_rise.init);
    log_info("syn_0_rise_value = %11.4k\n", parameters->syn_0_rise.synaptic_input_value);
    log_info("syn_0_decay = %R\n", (unsigned fract) parameters->syn_0.decay);
    log_info("syn_0_init  = %R\n", (unsigned fract) parameters->syn_0.init);
    log_info("syn_0_value = %11.4k\n", parameters->syn_0.synaptic_input_value);

    log_info("syn_1_rise_decay = %R\n", (unsigned fract) parameters->syn_1_rise.decay);
    log_info("syn_1_rise_init  = %R\n", (unsigned fract) parameters->syn_1_rise.init);
    log_info("syn_1_rise_value = %11.4k\n", parameters->syn_1_rise.synaptic_input_value);
    log_info("syn_1_decay = %R\n", (unsigned fract) parameters->syn_1.decay);
    log_info("syn_1_init  = %R\n", (unsigned fract) parameters->syn_1.init);
    log_info("syn_1_value = %11.4k\n", parameters->syn_1.synaptic_input_value);

    log_info("syn_2_rise_decay = %R\n", (unsigned fract) parameters->syn_2_rise.decay);
    log_info("syn_2_rise_init  = %R\n", (unsigned fract) parameters->syn_2_rise.init);
    log_info("syn_2_rise_value = %11.4k\n", parameters->syn_2_rise.synaptic_input_value);
    log_info("syn_2_decay = %R\n", (unsigned fract) parameters->syn_2.decay);
    log_info("syn_2_init  = %R\n", (unsigned fract) parameters->syn_2.init);
    log_info("syn_2_value = %11.4k\n", parameters->syn_2.synaptic_input_value);

    log_info("syn_3_rise_decay = %R\n", (unsigned fract) parameters->syn_3_rise.decay);
    log_info("syn_3_rise_init  = %R\n", (unsigned fract) parameters->syn_3_rise.init);
    log_info("syn_3_rise_value = %11.4k\n", parameters->syn_3_rise.synaptic_input_value);
    log_info("syn_3_decay = %R\n", (unsigned fract) parameters->syn_3.decay);
    log_info("syn_3_init  = %R\n", (unsigned fract) parameters->syn_3.init);
    log_info("syn_3_value = %11.4k\n", parameters->syn_3.synaptic_input_value);
}

#endif  // _SYNAPSE_TYPES_GLIF3_IMPL_H_
