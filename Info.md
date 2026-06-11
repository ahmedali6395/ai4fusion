# **NaN count per column:**

## **beta_n**

_\# of NaNs: 0_

"Normalized plasma β, (ratio between
plasma pressure and total magnetic
pressure). Concretely: β_n = β B_0 a/I_p
where B_0 [T] is the on-axis magnetic
field, a [m] is the minor radius, and I_p
[MA] is the plasma current. High β_n
correlates with pressure-driven
instabilities."

_Valid Range:_ [0.0, 2.0]

## **beta_p**

_\# of NaNs: 0_

Plasma poloidal β, defined as the ratio
between plasma pressure and poloidal
magnetic pressure.

_Valid Range:_ [0.0, 1.5]

## **chisq**

_\# of NaNs: 0_

Error between the EFIT result and the
magnetic probe measurements. A
smaller chisq represents a more
accurate fit of the plasma profile. Note:
you should first use chisq to filter out
rows which contain unreliable values.

_Valid Range:_ [0.1, 40.0]

## **dipprog_dt**

_\# of NaNs: 0_

Time derivative of the programmed
plasma current. Useful for identifying the
“flattop” period of the plasma

_Valid Range:_ [-5.0e6, 5.0e6]

## **greenwald_fraction**

_\# of NaNs: 0_

Line-averaged electron density (n_e
[1020 m3]) divided by the Greenwald
density (n_G [1020 m3] = I_p/πa^2)
The Greenwald density is an empirical
scaling defined in terms of plasma
current I_p [MA] and minor radius of
the plasma a. Greenwald_fraction > 1
tends to result in “density limit
disruptions”.

_Valid Range:_ [0.0, 1.5]

## **ip**

_\# of NaNs: 0_

Plasma current. Main feature in
describing the lifetime of the plasma. The
disruption is primarily defined with
respect to changes in this paramete

_Valid Range:_ N/A

## **ip_error**

_\# of NaNs: 0_

Deviation of the plasma current from its
pre-programmed value, (ip-ipprog).
Large errors suggest unexpected
changes in the plasma core.

_Valid Range:_ N/A

## **kappa**

_\# of NaNs: 0_

Plasma vertical elongation. High
elongations can lead to the plasma
becoming vertically unstable.

_Valid Range:_ [0.8, 2.0]

## **li**

_\# of NaNs: 0_

Plasma normalized internal inductance.
A scalar that correlates well with the
shape of the current-density profile.
li>1 corresponds to current density
peaked in the core, li~1 to a nearly
uniform current density, and li<1 to
current density peaked near the plasma
edge. The shape of the current-profile
strongly affects the strength of
current-driven instabilities.

_Valid Range:_ [0.2, 4.5]

## **lower_gap**

_\# of NaNs: 0_

Gap between bottom of plasma and the
machine. A small lower_gap can suggest
that the plasma is moving vertically down
and can cause the plasma to accumulate
more impurities.

_Valid Range:_ [0.025, 0.25]

## **n_e**

_\# of NaNs: 0_

Line-averaged electron density, as
measured from TCI chord 4 (generally
near the plasma core). Too high a
density can lead to hitting the Greenwald
density limit and/or excessive radiation,
while too low a density can lead to
locked modes.

_Valid Range:_ [0.2e20, 7e20]

## **n_equal_1_mode**

_\# of NaNs: 122426_

Amplitude of the n=1 component of the
perturbed magnetic field. Indicative of
the strength of MHD instabilities within
the plasma.

_Valid Range:_ [0.0, 0.02]

## **n_equal_1_normalized**

_\# of NaNs: 122426_

Amplitude of the n=1 component of the
perturbed magnetic field divided by the
toroidal magnetic field B_0. Proxy for
presence of locked modes, an important
disruption precursor.

_Valid Range:_ N/A

## **n_equal_1_phase**

_\# of NaNs: ?_

_Valid Range:_ ?

## **n_over_ncrit**

_\# of NaNs: 92_

Vertical stability parameter of the
plasma. When n_over_ncrit > ~1
the plasma is more prone to vertical
displacement event (VDE) disruptions.

_Valid Range:_ [-3.0, 3.0]

## **p_icrf**

_\# of NaNs: 0_

Heating power launched towards the
plasma via radio frequency ion cyclotron
waves. Increases in heating power can
significantly affect the operating
scenario, and sudden drops in heating
can lead to radiative collapse
disruptions.

_Valid Range:_ [0.0, inf]

## **p_lh**

_\# of NaNs: 0_

Heating power launched towards the
plasma via radio frequency lower hybrid
waves. Similar effect on disruptivity as
p_icrf.

_Valid Range:_ [0.0, inf]

## **p_oh**

_\# of NaNs: 0_

Heating power deposited in the plasma
due to resistive/ohmic heating of the
plasma. A more stable form of heating
than the other two, but still contributes to
the total power-balance in the plasma.

_Valid Range:_ [0.0, 5.0e6]

## **p_rad**

_\# of NaNs: 0_

Power radiated out of plasma. Excessive
radiation can cool the plasma and lead to
radiative collapse disruptions.

_Valid Range:_ [0.0, inf]

## **q0**

_\# of NaNs: 0_

Safety factor at the core plasma. Low
q0’s are indicative of very peaked
current-profiles which can lead to
sawteeth instabilities.

_Valid Range:_ [0.0, 2.0]

## **q95**

_\# of NaNs: 0_

Safety factor at the flux surface
corresponding to ψ95 = 0.95\*(ψedge -
ψmag-axis). Low q95 correlates with MHD
instabilities.

_Valid Range:_ [0.0, inf]

## **qstar**

_\# of NaNs: 0_

Safety factor at the edge of plasma,
approximating the plasma as a cylinder.
Similar consequences to q95.

_Valid Range:_ [0.0, inf]

## **radiated_fraction**

_\# of NaNs: 0_

The ratio of radiated power to input
power. radiated_fraction > 1
implies the plasma is radiating more
power than it’s absorbing, and therefore
often correlates with radiative collapse
disruptions.

_Valid Range:_ [0.0, 15.0]

## **shot**

_\# of NaNs: ?_

Unique discharge number. shot is
replicated for each time slice. The digits
in the 9-10 digit number correspond to
[1]YY/MM/DD/### of when the shot was
taken, where 1YY corresponds to 20YY.

_Valid Range:_ don't worry about it

## **ssep**

_\# of NaNs: 0_

Outboard radial distance to external
second separatrix for single null
configurations. positive for double nulls
or single top null and negative for single
bottom null.
https://cmodwiki.psfc.mit.edu/index.php/
EFIT_variables

_Valid Range:_ [-1.0, 1.0]

## **time**

_\# of NaNs: 0_

Time during the discharge.

_Valid Range:_ don't worry about it

## **time_until_disrupt**

_\# of NaNs: 353911_

Elapsing time before the disruption
event. This is your target variable.
time_until_disrupt=0 is the
disruption, time_until_disrupt =
t_dis-time. t_dis is not explicitly
given, but can be derived via the inverse
formula.

_Valid Range:_ NaN or numeric

## **upper_gap**

_\# of NaNs: 0_

Gap between top of plasma and the
machine. Similar implications to
lower_gap.

_Valid Range:_ [0.0, inf]

## **v_loop**

_\# of NaNs: 0_

Voltage across a full toroidal loop of the
plasma edge. Large changes in v_loop
are indicative of significant changes in
the plasma resistivity and other core
characteristics.

_Valid Range:_ [-7.0, 7.0]

## **wmhd**

_\# of NaNs: 0_

Total thermal energy stored in the
plasma. Effects power-balance within the
plasma.

_Valid Range:_ [0.0, 3.0e5]

## **z_error**

_\# of NaNs: 0_

Difference between the actual vertical
position of the current centroid of the
plasma and the programmed position.
Concretely, z_error = zcur -
z_prog. High z_error usually
proceeds VDE disruptions.

_Valid Range:_ [-1.0, 1.0]

## **z_cur**

_\# of NaNs: 0_

Actual vertical position of the current
centroid. Similar implications to z_error.

abs(zcur) should not be greater than the
plasma's vertical minor radius
(~a\*kappa~44cm). Use +-50cm just to be
safe. Ref:
https://cmodwiki.psfc.mit.edu/index.php/
Alcator_Parameters

_Valid Range:_ [-0.5, 0.5]

---

a_minor 0

bt 122426

dbetap_dt 0

dip_dt 0

dip_smoothed 0

dli_dt 0

dn_dt 0

dprad_dt 8

dwmhd_dt 0

i_efc 426

ip_prog 0

kappa_area 0

ne_peaking 123393

p_input 0

prad_peaking 153935

pressure_peaking 123393

rmagx 0

sxr 2463

tau_rad 555

te_core_vs_avg_ece 122804

te_edge_vs_avg_ece 122804

te_peaking 123393

te_width 99352

te_width_ece 122804

tribot 0

tritop 0

v_loop_efit 0

v_surf 0

v_z 0

z_prog 0

z_times_v_z 0

disruptive 0

dtype: int64

Reeces truncating data:
Rows remaining vs original: 424252/526251 (80.62%)
Columns remaining vs original: 51/65 (78.46%)
Entries remaining vs original: 21636852/34206315 (63.25%)
