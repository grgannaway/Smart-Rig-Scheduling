This is a smart-scheduler for generating riglines with project data and constraints.   This branch is unique from previous branch in that it layers in spreadsheet inputs for constraints of capital/production/FCF.  The objective function stays the same of maximizing NPV, but other constraints are honored as well.  Also, this new version has the following changes:

Mandatory dates are limit dates, not exact dates.
Allow for multiple mandatory dates for each constraint.
cap water production at a max rate per well (2000 bwpd)
drop rig threshold - must give x months before you're allowed to pick rig back up
must wait 21+ days between spud end and frac start, unless non-continuous frac then 50.
non continuous frac = 5+ days between frac jobs.
create duc days output
mandatory 30+ days between frac end and frac start if there needs to be a non-continuous frac (frac gap)
add price adder for winter months (via csv import) - replace price input with price csv
csv import for fixed topside volumes
production topside % as an adder to calculated base and wedge sans csv topside import.
simulate only after x date (may already work, as there's a simulation start date).  Just check logic against mandatory dates and how capex/prod works.




This is an algorithm shift to Genetic algorithms from linear programming.  GAs are better for searching non-linear domain spaces that cashflows/production from wells exhibit.  CP-SAT can use simplified assumptions to linearlize the data but it is not robust.  GA can be rigorous in their search space across the entire problem set.
