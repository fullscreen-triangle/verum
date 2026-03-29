# Philharmonic — F1 Telemetry Validation

Philharmonic validates the **vehicle circuit graph framework** using real Formula 1 telemetry data. It builds a 20-node circuit graph representing an F1 car's major subsystems, applies trajectory completion from partial observations (public telemetry channels), and validates against known F1 behaviour.

## Architecture

The F1 car is modelled as a 20-node circuit graph:

- **Power unit**: ICE, Turbocharger, MGU-K, MGU-H, Battery/ES
- **Drivetrain**: Gearbox, Differential
- **Wheels**: FL, FR, RL, RR
- **Brakes**: FL, FR, RL, RR (thermal nodes)
- **Suspension**: FL, FR, RL, RR
- **Aerodynamics**: Downforce (speed-dependent)

Observable nodes (public FastF1 channels): Speed, RPM, Throttle, Brake, Gear, DRS.
Hidden nodes (reconstructed via trajectory completion): Turbo, MGU-K/H, Battery, Differential, Suspensions.

## Validation Tests

### Test 1: State Reconstruction
Reconstruct hidden subsystem states from partial telemetry. Verify that turbo RPM tracks engine RPM, MGU-K correlates with braking/acceleration, battery SOC oscillates correctly, and suspension loads track downforce.

### Test 2: Fault Prediction
Detect component failures before they happen by tracking backward trajectories through state space. When a node's trajectory escapes the healthy attractor, the fault is flagged with a lead time measured in laps.

### Test 3: Tire Degradation
Predict the tire performance cliff by tracking tire node categorical depth over a stint. The wear-dependent conductance model captures the nonlinear degradation curve and identifies the cliff lap.

### Test 4: Racing Line
Extract the optimal racing line from S-entropy analysis. The fastest lap's S-entropy trace defines the optimal path in [S_k, S_t, S_e] space, compared with the aggregate molecular trail.

## Usage

```bash
cd verum
python -m philharmonic.validation.run_all
```

This runs all 4 validation tests and generates 5 publication-quality panels in `philharmonic/figures/`.

## Dependencies

- **fastf1** — Formula 1 telemetry access (with automatic synthetic fallback)
- **numpy** — numerical computation
- **scipy** — linear algebra, optimisation
- **matplotlib** — publication-quality figures

## Output

Five 300-DPI panels:
1. `panel_1_f1_circuit.png` — Circuit graph structure
2. `panel_2_reconstruction.png` — State reconstruction accuracy
3. `panel_3_fault.png` — Fault prediction and detection
4. `panel_4_tires.png` — Tire degradation and cliff prediction
5. `panel_5_racing_line.png` — Racing line S-entropy analysis
