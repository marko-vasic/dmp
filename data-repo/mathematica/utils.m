(* ::Package:: *)

softmax[x_] := SetPrecision[N@Exp[x]/Total[Exp[x],{-1}], 20];

EpsilonSatisfied[current_, target_, epsilon_] :=
  Module[{},
    Return[Abs[current - target] <= epsilon];
  ];

EpsilonPercentSatisfied[current_, target_, percent_] :=
  Module[{epsilon},
    epsilon = Abs[(1 - percent) * target];
    If[epsilon < 0.001, epsilon = 0.001];
    Return[EpsilonSatisfied[current, target, epsilon]];
  ];

EpsilonSatisfiedByAll[plotter_, targets_, tCurrent_, epsilon_] :=
  Module[{i, current},
    For[i = 1, i <= Length[plotter], i++,
      current = plotter[[i]] /. {t -> tCurrent};
      If[!EpsilonSatisfied[current, targets[[i]], epsilon], Return[False]];
    ];
    Return[True];
  ];

EpsilonPercentSatisfiedByAll[plotter_, targets_, tCurrent_, percent_] :=
  Module[{i, current},
    For[i = 1, i <= Length[plotter], i++,
      current = plotter[[i]] /. {t -> tCurrent};
      If[!EpsilonPercentSatisfied[current, targets[[i]], percent], Return[False]];
    ];
    Return[True];
  ];

OrderSatisfied[plotter_, targets_, tCurrent_] :=
  Module[{targetOrdering, currentOrdering, currentValues},
    targetOrdering = Ordering[targets];
    currentValues = plotter /. {t -> tCurrent};
    currentOrdering = Ordering[currentValues];
    Return[currentOrdering == targetOrdering];
  ];

AbsoluteConfidenceSatisfiedByAll[plotter_, targetConfidences_, tCurrent_, epsilon_] :=
  Module[{values, confidences, current, i},
    values = plotter /. {t -> tCurrent};
    confidences = softmax[values];
       
    For[i = 1, i <= Length[plotter], i++,
      current = confidences[[i]];
      If[!EpsilonSatisfied[current, targetConfidences[[i]], epsilon], Return[False]];
    ];
    Return[True];
  ];

FractionConfidenceSatisfiedByAll[plotter_, targetConfidences_, tCurrent_, percent_] :=
  Module[{values, confidences, current, i},
    values = plotter /. {t -> tCurrent};
    confidences = softmax[values];
       
    For[i = 1, i <= Length[plotter], i++,
      current = confidences[[i]];
      If[!EpsilonPercentSatisfied[current, targetConfidences[[i]], percent], Return[False]];
    ];
    Return[True];
  ];

EpsilonCompletionTime[function_, targetValue_, tmax_, epsilon_, tdelta_:0.01]:=
  (*
  Computes epsilon completion time for a function.
    - @param function is a function for which completion time is computed.
    - @param targetValue is a target completion value.
    - @param tmax is the time moment of end of the simulation (used to bound function time domain).
    - @param epsilon is the epsilon value (when function is in epsilon range of the targetVal it is considered converged.
  *)
  Module[{tCurrent, finalValue, currentValue},
    If[epsilon <= 0, 
       Throw["Invalid epsilon value -- epsilon should be positive number."]];
    finalValue = function /. {t->tmax};
    (* Did not converge to epsilon range. *)
    If[!EpsilonSatisfied[finalValue, targetValue, epsilon], Return[Null]];
    
    For[tCurrent=tmax, tCurrent > 0, tCurrent = tCurrent - tdelta,
      currentValue = function /. {t -> tCurrent};
      (* First time since the end of simulation when it got outside of the epsilon range *)
      If[!EpsilonSatisfied[currentValue, targetValue, epsilon], Return[tCurrent + tdelta]];
    ];
    (*
     It is not valid to have a precheck of whether value is in the epsilon range from the start
     since the value could have oscilated outside and inside of epsilon range and we want to find
     the last time moment when it was in the epsilon range and did not go outside of it again.
    *)
    Return[0];
  ];

EpsilonConvergenceTimes[plotter_, targets_, tmax_, epsilon_, tdelta_:0.01]:=
  (* Will return epsilon convergence times for each output.
     - @param plotter is a list with simulation results of different variables,
     - @param targets is a list with the final values that those variables should reach.
     - @tmax is the time moment of end of the simulation (used to bound function time domain).
     - @epsilon is the epsilon value (when function is in epsilon range of the targetVal it is considered converged.
     - @returns a list of convergence times
  *)
   Module[{notComputedCode, times, i, finalValues, currentValues, tCurrent},
    notComputedCode = -69;
    times = Table[notComputedCode, {Length[plotter]}];

    finalValues = plotter /. {t->tmax};
    For[i=1, i<=Length[plotter], i++,
      If[!EpsilonSatisfied[finalValues[[i]], targets[[i]], epsilon], times[[i]] = Null];
    ];

    For[tCurrent=tmax, tCurrent>0, tCurrent=tCurrent-tdelta,
      currentValues = plotter /. {t->tCurrent};
      For[i=1, i<=Length[plotter], i++,
        If[times[[i]]==notComputedCode && !EpsilonSatisfied[currentValues[[i]], targets[[i]], epsilon], times[[i]] = tCurrent + tdelta];
      ];
      If[!MemberQ[times, notComputedCode], Return[times]];
    ];

    For[i=1, i<=Length[plotter], i++,
      If[times[[i]]==notComputedCode, times[[i]] = 0];
    ];

    Return[times];
  ];

EpsilonPercentConvergenceTimes[plotter_, targets_, tmax_, targetPercent_, tdelta_:0.01]:=
  (* Will return epsilon convergence times for each output.
     - @param plotter is a list with simulation results of different variables,
     - @param targets is a list with the final values that those variables should reach.
     - @param tmax is the time moment of end of the simulation (used to bound function time domain).
     - @param targetPercent is the percentage of the targets value that should be reached to declare convergence.
     - @returns a list of convergence times
  *)
   Module[{notComputedCode, times, finalValues, currentValues, i},
    notComputedCode = -69;
    times = Table[notComputedCode, {Length[plotter]}];

    finalValues = plotter /. {t->tmax};
    For[i=1, i<=Length[plotter], i++,
      If[!EpsilonPercentSatisfied[finalValues[[i]], targets[[i]], targetPercent], times[[i]] = Null];
    ];

    For[tCurrent=tmax, tCurrent > 0, tCurrent = tCurrent - tdelta,
      currentValues = plotter /. {t->tCurrent};
      For[i=1, i<=Length[plotter], i++,
        If[times[[i]]==notComputedCode && !EpsilonPercentSatisfied[currentValues[[i]], targets[[i]], targetPercent], times[[i]] = tCurrent + tdelta];
      ];
      If[!MemberQ[times, notComputedCode], Return[times]];
    ];

    For[i=1, i<=Length[plotter], i++,
      If[times[[i]]==notComputedCode, times[[i]] = 0];
    ];

    Return[times];
  ];
  
AbsoluteConfidenceConvergenceTimes[plotter_, targets_, tmax_, epsilon_, tdelta_:0.01]:=
  (* Will return confidence convergence times for each output.
     - @param plotter is a list with simulation results of different variables,
     - @param targets is a list with the final values that those variables should reach.
     - @param tmax is the time moment of end of the simulation (used to bound function time domain).
     - @param epsilon is the range of the final confidence that should be reached to declare convergence.
     - @returns a list of convergence times
  *)
   Module[{notComputedCode, times, tCurrent, targetConfidences, currentConfidences, finalConfidences, i},
    notComputedCode = -69;
    times = Table[notComputedCode, {Length[plotter]}];

    targetConfidences = softmax[targets];
    finalConfidences = softmax[plotter /. {t->tmax}];
    For[i=1, i<=Length[plotter], i++,
      If[!EpsilonSatisfied[finalConfidences[[i]], targetConfidences[[i]], epsilon], times[[i]] = Null];
    ];

    For[tCurrent=tmax, tCurrent > 0, tCurrent = tCurrent - tdelta,
      currentConfidences = softmax[plotter /. {t->tCurrent}];
      For[i=1, i<=Length[plotter], i++,
        If[times[[i]]==notComputedCode && !EpsilonSatisfied[currentConfidences[[i]], targetConfidences[[i]], epsilon], times[[i]] = tCurrent + tdelta];
      ];
      If[!MemberQ[times, notComputedCode], Return[times]];
    ];

    For[i=1, i<=Length[plotter], i++,
      If[times[[i]]==notComputedCode, times[[i]] = 0];
    ];

    Return[times];
  ];

FractionConfidenceConvergenceTimes[plotter_, targets_, tmax_, percent_, tdelta_:0.01]:=
  (* Will return confidence convergence times for each output.
     - @param plotter is a list with simulation results of different variables,
     - @param targets is a list with the final values that those variables should reach.
     - @param tmax is the time moment of end of the simulation (used to bound function time domain).
     - @param epsilon is the range of the final confidence that should be reached to declare convergence.
     - @returns a list of convergence times
  *)
  Module[{notComputedCode, times, targetConfidences, finalConfidences, currentConfidences, i, tCurrent},
    notComputedCode = -69;
    times = Table[notComputedCode, {Length[plotter]}];

    targetConfidences = softmax[targets];
    finalConfidences = softmax[plotter /. {t->tmax}];
    For[i=1, i<=Length[plotter], i++,
      If[!EpsilonPercentSatisfied[finalConfidences[[i]], targetConfidences[[i]], percent], times[[i]] = Null];
    ];

    For[tCurrent=tmax, tCurrent > 0, tCurrent = tCurrent - tdelta,
      currentConfidences = softmax[plotter /. {t->tCurrent}];
      For[i=1, i<=Length[plotter], i++,
        If[times[[i]]==notComputedCode && !EpsilonPercentSatisfied[currentConfidences[[i]], targetConfidences[[i]], percent], times[[i]] = tCurrent + tdelta];
      ];
      If[!MemberQ[times, notComputedCode], Return[times]];
    ];

    For[i=1, i<=Length[plotter], i++,
      If[times[[i]]==notComputedCode, times[[i]] = 0];
    ];

    Return[times];
  ];

OrderConvergenceTime[plotter_, targets_, tmax_, tdelta_:0.01]:=
  (* Returns the time moment when output signals reach their final ordering. *)
  Module[{finalValues, targetOrdering, finalOrdering, tCurrent, currentOrdering},
    targetOrdering = Ordering[targets];
    
    finalValues = plotter /. {t->tmax};
    finalOrdering = Ordering[finalValues];
    (* System did not converge *)
    If[finalOrdering != targetOrdering, Return[Null]];
    
    For[tCurrent=tmax, tCurrent > 0, tCurrent = tCurrent - tdelta,
      currentOrdering = Ordering[plotter /. {t -> tCurrent}];
      (* First time since the end of simulation when it got outside of the epsilon range *)
      If[currentOrdering != targetOrdering, Return[tCurrent + tdelta]];
    ];
    Return[0];
  ];

OrderOfHighestConvergenceTime[plotter_, targets_, tmax_, tdelta_:0.01]:=
  (* Returns a time moment when signal which is eventually going to be the highest is above other signals. *)
  Module[{finalValues, targetOrdering, finalOrdering, tCurrent, currentOrdering},
    targetOrdering = Reverse[Ordering[targets]];
    
    finalValues = plotter /. {t->tmax};
    finalOrdering = Reverse[Ordering[finalValues]];
    (* System did not converge *)
    If[finalOrdering[[1]] != targetOrdering[[1]], Return[Null]];
    
    For[tCurrent=tmax, tCurrent > 0, tCurrent = tCurrent - tdelta,
      currentOrdering = Reverse[Ordering[plotter /. {t -> tCurrent}]];
      (* First time since the end of simulation when it got outside of the epsilon range *)
      If[currentOrdering[[1]] != targetOrdering[[1]], Return[tCurrent + tdelta]];
    ];
    Return[0];
  ];

AnalyzeConvergence[plotter_, targets_, correctOutput_, tmax_]:=
  (*
    - @returns A list containing different types of convergence times
        1. epsilon (e=0.01) convergence time of the slowest converging output (aka maximum epsilon convergence time).
        2. epsilon (e=0.01) convergence time of the correct output.
        3. epsilon (e=0.05) convergence time of the slowest converging output (aka maximum epsilon convergence time).
        4. epsilon (e=0.05) convergence time of the correct output.
        5. epsilon (e=0.1) convergence time of the slowest converging output (aka maximum epsilon convergence time).
        6. epsilon (e=0.1) convergence time of the correct output.
        7. epsilon percent (.8) convergence time of the slowest converging output (aka maximum epsilon convergence time).
        8. epsilon percent (.8) convergence time of the correct output.
        9. epsilon percent (.9) convergence time of the slowest converging output (aka maximum epsilon convergence time).
        10. epsilon percent (.9) convergence time of the correct output.
        11. epsilon percent (.95) convergence time of the slowest converging output (aka maximum epsilon convergence time).
        12. epsilon percent (.95) convergence time of the correct output.
        13. epsilon percent (.99) convergence time of the slowest converging output (aka maximum epsilon convergence time).
        14. epsilon percent (.99) convergence time of the correct output.
        15. confidence convergence time (when a slowest output is within 1 of the final confidence).
        16. confidence convergence time (when the correct output is within 1 of the final confidence).
        17. confidence convergence time (when a slowest output is within 95% of the final confidence).
        18. confidence convergence time (when the correct output is withing 95% of the final confidence).
        19. order convergence time (time when signals are in the correct order).
        20. order of highest convergence time (time when the eventually highest signal reaches the highest position).
  *)
    (* TODO: Change that this function accepts list of configs of times to be computed. *)
    Module[{epsilonConvergenceTimes, epsilonPercentConvergenceTimes, results, 
          confidences, correctOutputConfidence, confidenceConvergenceTime,
          epsilonRange, orderConvergenceTime},
    results = List[];

    epsilonConvergenceTimes = EpsilonConvergenceTimes[plotter, targets, tmax, 0.01];
    AppendTo[results, If[MemberQ[epsilonConvergenceTimes, Null], Null, Max[epsilonConvergenceTimes]]];
    AppendTo[results, epsilonConvergenceTimes[[correctOutput]]];
      
    epsilonConvergenceTimes = EpsilonConvergenceTimes[plotter, targets, tmax, 0.05];
    AppendTo[results, If[MemberQ[epsilonConvergenceTimes, Null], Null, Max[epsilonConvergenceTimes]]];
    AppendTo[results, epsilonConvergenceTimes[[correctOutput]]];

    epsilonConvergenceTimes = EpsilonConvergenceTimes[plotter, targets, tmax, 0.1];
    AppendTo[results, If[MemberQ[epsilonConvergenceTimes, Null], Null, Max[epsilonConvergenceTimes]]];
    AppendTo[results, epsilonConvergenceTimes[[correctOutput]]];

    epsilonPercentConvergenceTimes = EpsilonPercentConvergenceTimes[plotter, targets, tmax, 0.8];
    AppendTo[results, If[MemberQ[epsilonPercentConvergenceTimes, Null], Null, Max[epsilonConvergenceTimes]]];
    AppendTo[results, epsilonPercentConvergenceTimes[[correctOutput]]];

    epsilonPercentConvergenceTimes = EpsilonPercentConvergenceTimes[plotter, targets, tmax, 0.9];
    AppendTo[results, If[MemberQ[epsilonPercentConvergenceTimes, Null], Null, Max[epsilonConvergenceTimes]]];
    AppendTo[results, epsilonPercentConvergenceTimes[[correctOutput]]];

    epsilonPercentConvergenceTimes = EpsilonPercentConvergenceTimes[plotter, targets, tmax, 0.95];
    AppendTo[results, If[MemberQ[epsilonPercentConvergenceTimes, Null], Null, Max[epsilonConvergenceTimes]]];
    AppendTo[results, epsilonPercentConvergenceTimes[[correctOutput]]];

    epsilonPercentConvergenceTimes = EpsilonPercentConvergenceTimes[plotter, targets, tmax, 0.99];
    AppendTo[results, If[MemberQ[epsilonPercentConvergenceTimes, Null], Null, Max[epsilonConvergenceTimes]]];
    AppendTo[results, epsilonPercentConvergenceTimes[[correctOutput]]];

    confidenceConvergenceTimes = AbsoluteConfidenceConvergenceTimes[plotter, targets, tmax, 0.01];
    AppendTo[results, If[MemberQ[confidenceConvergenceTimes, Null], Null, Max[epsilonConvergenceTimes]]];
    AppendTo[results, confidenceConvergenceTimes[[correctOutput]]];

    confidenceConvergenceTimes = FractionConfidenceConvergenceTimes[plotter, targets, tmax, 0.95];
    AppendTo[results, If[MemberQ[confidenceConvergenceTimes, Null], Null, Max[epsilonConvergenceTimes]]];
    AppendTo[results, confidenceConvergenceTimes[[correctOutput]]];

    orderConvergenceTime = OrderConvergenceTime[plotter, targets, tmax];
    AppendTo[results, orderConvergenceTime];

    orderConvergenceTime = OrderOfHighestConvergenceTime[plotter, targets, tmax];
    AppendTo[results, orderConvergenceTime];

    Return[results];
  ];


ComputeConvergenceMultiple[plotter_, targets_, tmax_, configs_, tdelta_: 0.01] :=
  (*
    Computes different types of convergence times specified by the configs.
  *)
  (* TODO: Do performance comparison with the AnalyzeConvergence function for the same set of convergence times computed. *)
  Module[{notComputedCode, results, targetConfidences,
          i, convergenceType, tCurrent, converged},
    notComputedCode = -69;
    results = Table[notComputedCode, {Length[configs]}];

    targetConfidences = softmax[targets];

    For[i=1, i <= Length[configs], i++,
      convergenceType = configs[[i]][[1]];
      Switch[convergenceType,
             "E", If[!EpsilonSatisfiedByAll[plotter, targets, tmax, configs[[i]][[2]]],
                     results[[i]] = Null],
             "F", If[!EpsilonPercentSatisfiedByAll[plotter, targets, tmax, configs[[i]][[2]]],
                     results[[i]] = Null],
             "CE", If[!AbsoluteConfidenceSatisfiedByAll[plotter, targetConfidences, tmax, configs[[i]][[2]]],
                      results[[i]] = Null],
             "CF", If[!FractionConfidenceSatisfiedByAll[plotter, targetConfidences, tmax, configs[[i]][[2]]],
                      results[[i]] = Null],
             "O", If[!OrderSatisfied[plotter, targets, tmax],
                     results[[i]] = Null]
      ];
    ];
       
    For[tCurrent = tmax, tCurrent > 0, tCurrent = tCurrent - tdelta,
      For[i=1, i <= Length[configs], i++,
        If[results[[i]] == Null || results[[i]] != notComputedCode, Continue[]];
        convergenceType = configs[[i]][[1]];
        converged = False;
        Switch[convergenceType,
               "E", If[!EpsilonSatisfiedByAll[plotter, targets, tCurrent, configs[[i]][[2]]],
                       converged = True],
               "F", If[!EpsilonPercentSatisfiedByAll[plotter, targets, tCurrent, configs[[i]][[2]]],
                       converged = True],
               "CE", If[!AbsoluteConfidenceSatisfiedByAll[plotter, targetConfidences, tCurrent, configs[[i]][[2]]],
                       converged = True],
               "CF", If[!FractionConfidenceSatisfiedByAll[plotter, targetConfidences, tCurrent, configs[[i]][[2]]],
                       converged = True],
               "O", If[!OrderSatisfied[plotter, targets, tCurrent],
                       converged = True]
        ];
        If[converged, results[[i]] = tCurrent + tdelta];
      ];
      If[!MemberQ[results, notComputedCode], Return[results]];
    ];
       
    For[i = 1, i <= Length[results], i++,
      If[results[[i]] == notComputedCode, results[[i]] = 0];
    ];

    Return[results];
  ];

ComputeConvergenceRandomNNs[plotter_, targets_, tmax_, tdelta_: 0.01] :=
   Module[{},
     Return[ComputeConvergenceMultiple[
              plotter,
              correctResults,
              tmax,
              {{"E",0.07},{"F",0.9},{"CE",0.01},{"CF",0.95},{"O",Null}},
              tdelta=tdelta]
         ];
   ];
