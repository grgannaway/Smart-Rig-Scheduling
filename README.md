this is a smart-scheduler for generating riglines with project data and constraints.   This branch is unique from previous branch in that it incorporate water constraints.  

Also, this is an algorithm shift to Genetic algorithms.  GAs are better for searching non-linear domain spaces that cashflows/production from wells exhibit.  CP-SAT can use simplified assumptions to linearlize the data but it is not robust.  GA can be rigorous in their search space across the entire problem set.
