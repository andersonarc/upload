#ifndef _NEURON_MODEL_GLIF3_IMPL_H_
#define _NEURON_MODEL_GLIF3_IMPL_H_

#include <neuron/models/neuron_model.h>
#include <common/maths-util.h>
#include <debug.h>

/**
 * GLIF3 (Generalized Leaky Integrate-and-Fire) Neuron Model
 * This model includes after-spike currents with exponential decay.
 *
 * Model equations:
 * dV/dt = (1/C_m) * [I_e + I_asc_0 + I_asc_1 - g*(V - E_L)]
 * dI_asc_j/dt = -k_j * I_asc_j, for j = 0, 1
 *
 * When V >= V_thresh, spike and reset:
 * - V -> V_reset
 * - I_asc_j -> I_asc_j * exp(-k_j * t_ref) + asc_amp_j
 */

// Neuron parameters structure - must match Python model exactly
struct neuron_params_t {
    // State variables (initial values)
    REAL V_init;              // Initial membrane voltage (mV)
    REAL I_asc_0_init;        // Initial after-spike current 0 (nA)
    REAL I_asc_1_init;        // Initial after-spike current 1 (nA)

    // Model parameters
    REAL C_m;                 // Membrane capacitance (nF)
    REAL E_L;                 // Resting potential (mV)
    REAL V_reset;             // Reset voltage (mV)
    REAL V_thresh;            // Threshold voltage (mV)
    REAL asc_amp_0;           // After-spike current amplitude 0 (nA)
    REAL asc_amp_1;           // After-spike current amplitude 1 (nA)
    REAL g;                   // Conductance (uS)
    REAL k0;                  // ASC decay rate 0 (1/ms)
    REAL k1;                  // ASC decay rate 1 (1/ms)
    REAL t_ref;               // Refractory period (ms)
    REAL I_offset;            // Offset current (nA)
};

// Neuron state structure
struct neuron_t {
    // State variables
    REAL V;                   // Membrane voltage (mV)
    REAL I_asc_0;             // After-spike current 0 (nA)
    REAL I_asc_1;             // After-spike current 1 (nA)

    // Model parameters (copied for efficiency)
    REAL C_m;                 // Membrane capacitance (nF)
    REAL E_L;                 // Resting potential (mV)
    REAL V_reset;             // Reset voltage (mV)
    REAL V_thresh;            // Threshold voltage (mV)
    REAL asc_amp_0;           // After-spike current amplitude 0 (nA)
    REAL asc_amp_1;           // After-spike current amplitude 1 (nA)
    REAL g;                   // Conductance (uS)
    REAL k0;                  // ASC decay rate 0 (1/ms)
    REAL k1;                  // ASC decay rate 1 (1/ms)
    REAL t_ref;               // Refractory period (ms)
    REAL I_offset;            // Offset current (nA)

    // Precomputed values for efficiency
    REAL exp_k0_dt;           // exp(-k0 * dt)
    REAL exp_k1_dt;           // exp(-k1 * dt)
    REAL exp_k0_tref;         // exp(-k0 * t_ref)
    REAL exp_k1_tref;         // exp(-k1 * t_ref)
    REAL dt_over_cm;          // dt / C_m (unused with exp integration)
    REAL g_dt_over_cm;        // g * dt / C_m (unused with exp integration)

    // Exponential Euler integration (matches TensorFlow)
    REAL v_decay;             // exp(-dt * g / C_m)
    REAL current_factor;      // (1 - v_decay) / g
    REAL reset_current;       // (V_reset - V_thresh) for soft reset

    // Spike tracking for TensorFlow-style reset
    uint32_t spiked_last_step; // 1 if spike occurred in previous timestep, 0 otherwise

    // Refractory period tracking
    uint32_t refract_timer;   // Refractory period timer (in time steps)
    uint32_t refract_steps;   // Total refractory period (in time steps)
};

static inline void neuron_model_initialise(neuron_t *state, neuron_params_t *params,
        uint32_t n_steps_per_timestep) {

    // Initialize state variables
    state->V = params->V_init;
    state->I_asc_0 = params->I_asc_0_init;
    state->I_asc_1 = params->I_asc_1_init;

    // Copy parameters
    state->C_m = params->C_m;
    state->E_L = params->E_L;
    state->V_reset = params->V_reset;
    state->V_thresh = params->V_thresh;
    state->asc_amp_0 = params->asc_amp_0;
    state->asc_amp_1 = params->asc_amp_1;
    state->g = params->g;
    state->k0 = params->k0;
    state->k1 = params->k1;
    state->t_ref = params->t_ref;
    state->I_offset = params->I_offset;

    // Precompute exponentials and other constants for efficiency
    // dt is the simulation timestep in milliseconds
    REAL dt = 1.0k / (REAL) n_steps_per_timestep;

    state->exp_k0_dt = expk(-state->k0 * dt);
    state->exp_k1_dt = expk(-state->k1 * dt);
    state->exp_k0_tref = expk(-state->k0 * state->t_ref);
    state->exp_k1_tref = expk(-state->k1 * state->t_ref);
    state->dt_over_cm = dt / state->C_m;
    state->g_dt_over_cm = state->g * dt / state->C_m;

    // Exponential Euler integration (matches TensorFlow line 171-172)
    // tau = C_m / g, decay = exp(-dt / tau) = exp(-dt * g / C_m)
    REAL tau = state->C_m / state->g;
    state->v_decay = expk(-dt / tau);
    // current_factor = (1/C_m) * (1 - decay) * tau = (1 - decay) / g
    state->current_factor = (ONE - state->v_decay) / state->g;
    // reset_current = (V_reset - V_thresh) applied when spike occurs (TensorFlow line 328)
    state->reset_current = state->V_reset - state->V_thresh;

    // Initialize spike tracking
    state->spiked_last_step = 0;

    // Calculate refractory period in time steps
    state->refract_steps = (uint32_t) (state->t_ref * n_steps_per_timestep);
    state->refract_timer = 0;
}

static inline void neuron_model_save_state(neuron_t *state, neuron_params_t *params) {
    // Save state variables for continuation in next run
    params->V_init = state->V;
    params->I_asc_0_init = state->I_asc_0;
    params->I_asc_1_init = state->I_asc_1;
}

static state_t neuron_model_state_update(
        uint16_t num_excitatory_inputs, const input_t* exc_input,
        uint16_t num_inhibitory_inputs, const input_t* inh_input,
        input_t external_bias, REAL current_offset, neuron_t *restrict neuron) {

    // Sum excitatory and inhibitory inputs
    REAL total_exc = ZERO;
    REAL total_inh = ZERO;
    for (uint32_t i = 0; i < num_excitatory_inputs; i++) {
        total_exc += exc_input[i];
    }
    for (uint32_t i = 0; i < num_inhibitory_inputs; i++) {
        total_inh += inh_input[i];
    }

    // Total input current including after-spike currents
    // IMPORTANT: Use current ASC values (before decay) per Allen Institute implementation
    REAL I_total = total_exc - total_inh + external_bias + neuron->I_offset + current_offset
                   + neuron->I_asc_0 + neuron->I_asc_1;

    // Update membrane voltage using exponential Euler integration (matches TensorFlow)
    // TensorFlow line 330-334:
    // new_v = v * decay + current_factor * (input_current + asc_1 + asc_2 + g*E_L) + reset_current
    // NOTE: Voltage updates even during refractory period (matches TensorFlow line 321-334)
    REAL g_times_EL = neuron->g * neuron->E_L;
    neuron->V = neuron->V * neuron->v_decay +
                neuron->current_factor * (I_total + g_times_EL);

    // Add reset current if spike occurred last timestep (TensorFlow line 328, 334)
    // reset_current = prev_z * (v_reset - v_th)
    if (neuron->spiked_last_step) {
        neuron->V += neuron->reset_current;
    }

    // Update after-spike currents (exponential decay) - TensorFlow line 325-326
    // new_asc = exp(-dt * k) * asc + prev_z * asc_amp
    neuron->I_asc_0 *= neuron->exp_k0_dt;
    neuron->I_asc_1 *= neuron->exp_k1_dt;

    // Add amplitude if spike occurred last timestep (uses prev_z in TensorFlow)
    if (neuron->spiked_last_step) {
        neuron->I_asc_0 += neuron->asc_amp_0;
        neuron->I_asc_1 += neuron->asc_amp_1;
        neuron->spiked_last_step = 0;  // Clear flag after processing
    }

    // Handle refractory period (TensorFlow line 341)
    // Voltage still updates, but return value prevents spike detection
    if (neuron->refract_timer > 0) {
        neuron->refract_timer--;
        // Return voltage well below threshold to prevent spiking during refractory
        // (matches TensorFlow: new_z = tf.where(new_r > 0., tf.zeros_like(new_z), new_z))
        return neuron->V_reset;
    }

    // Return voltage for threshold comparison
    return neuron->V;
}

static state_t neuron_model_get_membrane_voltage(const neuron_t *neuron) {
    return neuron->V;
}

static void neuron_model_has_spiked(neuron_t *restrict neuron) {
    // Mark that spike occurred for next timestep's reset current and ASC updates
    // (TensorFlow uses prev_z for reset and ASC amplitude addition)
    neuron->spiked_last_step = 1;

    // Start refractory period (TensorFlow line 321)
    neuron->refract_timer = neuron->refract_steps;

    // NOTE: Voltage and ASCs are NOT modified here. Instead:
    // - Reset current is applied in next timestep's state_update (line 169-170)
    // - ASC amplitude is added in next timestep's state_update (line 179-182)
    // This matches TensorFlow's use of prev_z for delayed reset and ASC updates
}

static inline void neuron_model_print_state_variables(const neuron_t *neuron) {
    log_info("V = %11.4k mV", neuron->V);
    log_info("I_asc_0 = %11.4k nA", neuron->I_asc_0);
    log_info("I_asc_1 = %11.4k nA", neuron->I_asc_1);
    log_info("Refract timer = %u", neuron->refract_timer);
}

static inline void neuron_model_print_parameters(const neuron_t *neuron) {
    log_info("C_m = %11.4k nF", neuron->C_m);
    log_info("E_L = %11.4k mV", neuron->E_L);
    log_info("V_reset = %11.4k mV", neuron->V_reset);
    log_info("V_thresh = %11.4k mV", neuron->V_thresh);
    log_info("asc_amp_0 = %11.4k nA", neuron->asc_amp_0);
    log_info("asc_amp_1 = %11.4k nA", neuron->asc_amp_1);
    log_info("g = %11.4k uS", neuron->g);
    log_info("k0 = %11.4k 1/ms", neuron->k0);
    log_info("k1 = %11.4k 1/ms", neuron->k1);
    log_info("t_ref = %11.4k ms", neuron->t_ref);
    log_info("I_offset = %11.4k nA", neuron->I_offset);
}

#endif // _NEURON_MODEL_GLIF3_IMPL_H_
