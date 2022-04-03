#!/usr/bin/env wolframscript

Get[Directory[] <> "/CRNSimulator.m"];

IRIS[x0SposInitial_,x0SnegInitial_,x1SposInitial_,x1SnegInitial_,x2SposInitial_,x2SnegInitial_,x3SposInitial_,x3SnegInitial_] :=
{
conc[x0Spos, x0SposInitial],
conc[x0Sneg, x0SnegInitial],
conc[x1Spos, x1SposInitial],
conc[x1Sneg, x1SnegInitial],
conc[x2Spos, x2SposInitial],
conc[x2Sneg, x2SnegInitial],
conc[x3Spos, x3SposInitial],
conc[x3Sneg, x3SnegInitial],
rxn[x1Spos,iL0N2Sneg,1],
rxn[x1Sneg,iL0N2Spos,1],
rxn[x2Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos,1],
rxn[x2Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg,1],
rxn[x3Spos,iL0N0Spos+iL0N1Spos+iL0N2Spos,1],
rxn[x3Sneg,iL0N0Sneg+iL0N1Sneg+iL0N2Sneg,1],
conc[iL0N0Sneg,0.7749611705],
conc[iL0N1Sneg,0.9962031582],
conc[iL0N2Spos,0.5672149702],
rxn[iL0N0Spos,mL0N0+hL0N0Spos,1],
rxn[mL0N0+iL0N0Sneg,hL0N0Sneg,1],
rxn[iL0N1Spos,mL0N1+hL0N1Spos,1],
rxn[mL0N1+iL0N1Sneg,hL0N1Sneg,1],
rxn[iL0N2Spos,mL0N2+hL0N2Spos,1],
rxn[mL0N2+iL0N2Sneg,hL0N2Sneg,1],
rxn[hL0N0Spos,hL1N0Sneg+hL1N1Sneg+hL1N2Spos,1],
rxn[hL0N0Sneg,hL1N0Spos+hL1N1Spos+hL1N2Sneg,1],
rxn[hL0N1Spos,hL1N0Sneg+hL1N1Sneg+hL1N2Spos,1],
rxn[hL0N1Sneg,hL1N0Spos+hL1N1Spos+hL1N2Sneg,1],
rxn[hL0N2Spos,hL1N0Sneg+hL1N1Spos+hL1N2Spos,1],
rxn[hL0N2Sneg,hL1N0Spos+hL1N1Sneg+hL1N2Sneg,1],
conc[hL1N0Spos,0.5796289994],
conc[hL1N1Sneg,0.4018247924],
conc[hL1N2Sneg,1.1487628737]
}

cnt = Count[IRIS[x0SposInitial,x0SnegInitial,x1SposInitial,x1SnegInitial,x2SposInitial,x2SnegInitial,x3SposInitial,x3SnegInitial], rxn[___]];
Print["Number of reactions: " <> ToString[cnt]];

tmax=50;

rsys=IRIS[0.674501,0,0.337848,0,0.421564,0,0.396172,0];
sol=SimulateRxnsys[rsys,tmax, MaxSteps->Automatic, AccuracyGoal->Automatic, Method->{"EquationSimplification"->"Residual"}];
results={(hL1N0Spos[tmax] - hL1N0Sneg[tmax])/.sol,(hL1N1Spos[tmax] - hL1N1Sneg[tmax])/.sol,(hL1N2Spos[tmax] - hL1N2Sneg[tmax])/.sol};
plotter={(hL1N0Spos[t] - hL1N0Sneg[t])/.sol,(hL1N1Spos[t] - hL1N1Sneg[t])/.sol,(hL1N2Spos[t] - hL1N2Sneg[t])/.sol};
p51=Plot[plotter, {t, 0, tmax}, PlotLabels->{"y0","y1","y2"}, PlotRange->All, AxesLabel->{"time","value"}];

p51=Plot[plotter, {t, 0, tmax},
        PlotLabels->{"setosa","versicolor","virginica"},
        PlotRange->All,
        AxesLabel -> {Style["time", Bold, 20], Style["value", Bold, 20]},
        LabelStyle -> {FontSize->14},
        PlotStyle->{{RGBColor["#005f86"], Thickness[0.015]},
                    {RGBColor["#bf5700"], Thickness[0.015]},
                    {RGBColor["#579d42"], Thickness[0.015]}}

        ];

Export["./IRIS_custom.pdf", p51];