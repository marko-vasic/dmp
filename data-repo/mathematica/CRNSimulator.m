(* ::Package:: *)

(* ::Text:: *)
(*Chemical Reaction Network (CRN) Simulator package is developed by David Soloveichik. Copyright 2009-2015. *)
(*http://users.ece.utexas.edu/~soloveichik/crnsimulator.html*)


(* ::Section:: *)
(*Public interface specification*)


BeginPackage["CRNSimulator`"];


rxn::usage="Represents an irreversible reaction. eg. rxn[a+b,c,1]";
revrxn::usage="Represents a reversible reaction. eg. revrxn[a+b,c,1,1]";
conc::usage="Initial concentration: conc[x,10] or conc[{x,y},10].";
term::usage="Represents an additive term in the ODE for species x. \
Species concentrations must be expressed in x[t] form. eg. term[x, -2 x[t]*y[t]]";


SimulateRxnsys::usage=
"SimulateRxnsys[rxnsys,endtime] simulates the reaction system rxnsys for time 0 \
to endtime. In rxnsys, initial concentrations are specified by conc statements. \
If no initial condition is set for a species, its initial concentration is set to 0. \
Rxnsys can also include term[] statements (e.g. term[x, -2 x[t]]) which are additively \
combined together with term[]s derived from rxn[] statements. \
Rxnsys can also include direct ODE definitions for some species (e.g. x'[t]==...), \
or direct definitions of species as functions of time (e.g. x[t]==...), \
which are passed to NDSolve without modification. \
Any options specified (eg WorkingPrecision->30) \
are passed to NDSolve."; 
SpeciesInRxnsys::usage=
"SpeciesInRxnsys[rxnsys] returns the species in reaction system rxnsys. \
SpeciesInRxnsys[rxnsys,pttrn] returns the species in reaction system rxnsys \
matching Mathematica pattern pttrn (eg x[1,_]).";
SpeciesInRxnsysStringPattern::usage=
"SpeciesInRxnsysPattern[rxnsys,pttrn] returns the species in reaction system rxnsys \
matching Mathematica string pattern pttrn. \
(Eg \"g$*\" matches all species names starting with \"g$\" ; \ 
can also do RegularExpression[\"o..d.\$.*\"].)";
RxnsysToOdesys::usage=
"RxnsysToOdesys[rxnsys,t] returns the ODEs corresponding to reaction system rxnsys, \
with initial conditions. If no initial condition is set for a species, its initial \
concentration is set to 0. \
The time variable is given as the second argument; if omitted it is set to Global`t.";

reduceFFNCCrn::usage=""
printProductsStats::usage=""
normalizeCRN::usage=""

(*To use instead of Sequence in functions with Hold attribute but not HoldSequence,
like Module, If, etc*)
Seq:=Sequence 


(* ::Section:: *)
(*Private*)


Begin["`Private`"];




(*We want rxn[a+b,c,1] to be different from rxn[b+a,c,1], so we have to set attribute
HoldAll. But we also want to evaluate if any variables can be evaluated.*)
(* SetAttributes[{rxn,revrxn}, HoldAll] *)
(* rxn[rs_Plus,ps_,k_]:=
 ReleaseHold[ReplacePart[rxn[1,ps,k],1->Hold[Plus]@@List@@Unevaluated[rs]]]/;
 Hold@@Unevaluated[rs] =!= Hold@@List@@Unevaluated[rs]
rxn[rs_,ps_Plus,k_]:=
 ReleaseHold[ReplacePart[rxn[rs,1,k],2->Hold[Plus]@@List@@Unevaluated[ps]]]/;
 Hold@@Unevaluated[ps] =!= Hold@@List@@Unevaluated[ps]
rxn[rs_,ps_,k_]:=
 (With[{rse=rs},rxn[rse,ps,k]])/;Head[rs]=!=Plus&&Unevaluated[rs]=!=rs
rxn[rs_,ps_,k_]:=
 (With[{pse=ps},rxn[rs,pse,k]])/;Head[ps]=!=Plus&&Unevaluated[ps]=!=ps
rxn[rs_,ps_,k_]:=
 (With[{ke=k},rxn[rs,ps,ke]])/;Unevaluated[k]=!=k *)


(* Automatically expand revrxn and concentration lists *)
revrxn[r_,p_,k1_,k2_]:=Sequence[rxn[r,p,k1],rxn[p,r,k2]]
conc[xs_List,c_]:=Seq@@(conc[#,c]&/@xs)

(* People are often confused about rxn[0,x,1] instead of rxn[1,x,1]. So we automatically replace any integer with 1. *)
rxn[Except[1,_Integer],ps_,k_]:=rxn[1,ps,k]


(* Species as products or reactants in rxn[] statements, as well as defined in x'[t]== or x[t]== statements \
or term statements, or conc statements *)
SpeciesInRxnsys[rxnsys_]:=
 Union[
 	Cases[Cases[rxnsys,rxn[r_,p_,_]:>Seq[r,p]]/.Times|Plus->Seq,s_Symbol|s_Symbol[__]],
 	Cases[rxnsys, x_'[_]==_ | x_[_]==_ | term[x_,__] | conc[x_,_] :> x]]
SpeciesInRxnsys[rsys_,pattern_]:=Cases[SpeciesInRxnsys[rsys],pattern]
SpeciesInRxnsysStringPattern[rsys_,pattern_]:=Select[SpeciesInRxnsys[rsys],StringMatchQ[ToString[#],pattern]&]


(* Check if a species' initial value is set in a odesys *) 	
InitialValueSetQ[odesys_,x_]:=
 MemberQ[odesys,x[_]==_]

(* Check if a species is missing an ODE or a direct definition (x[t]=_) in odesys. *)
MissingODEQ[odesys_,x_,t_Symbol]:=
 !MemberQ[odesys, D[x[t],t]==_ | x[t]==_]  	

RxnsysToOdesys[rxnsys_,t_Symbol:Global`t]:=
 Module[
  {spcs=SpeciesInRxnsys[rxnsys], concs, termssys, odesys, eqsFromTerms, eqsFromConcs},

  (* extract conc statements and sum them for same species *)
  concs = conc[#[[1,1]],Total[#[[;;,2]]]]&/@GatherBy[Cases[rxnsys,conc[__]],Extract[{1}]];

  (* Convert rxn[] to term[] statements *)
  termssys=rxnsys /. rxn:rxn[__]:>Seq@@ProcessRxnToTerms[rxn,t];

  (* ODEs from parsing terms *)
  eqsFromTerms = ProcessTermsToOdes[Cases[termssys,term[__]],t]; 
  (* initial values from parsing conc statements *)
  eqsFromConcs = Cases[concs,conc[x_,c_]:>x[0]==c]; 

  (* Remove term and conc statements from rxnsys and add eqs generated from them. 
     If there is a conflict, use pass-through equations *)
  odesys = DeleteCases[termssys, term[__]|conc[__]];
  odesys = Join[odesys,
                DeleteCases[eqsFromTerms, Alternatives@@(#'[t]==_& /@ Cases[odesys,(x_'[t]|x_[t])==_:>x])],
                eqsFromConcs];
     
  (* For species still without initial values, add zeros *)
  odesys = Join[odesys, #[0]==0& /@ Select[spcs, !InitialValueSetQ[odesys,#]&]];

  (* For species still without ODE or direct definition, add zero time derivative *)
  (* This can happen for example if conc is defined, but nothing else *)
  Join[odesys, D[#[t],t]==0& /@ Select[spcs, MissingODEQ[odesys,#,t]&]]
 ]
 
(* Create list of ODEs from parsing term statements. terms should be list of term[] statements. *) 
ProcessTermsToOdes[terms_,t_Symbol]:=
Module[{spcs=Union[Cases[terms,term[s_,_]:>s]]},
#'[t]==Total[Cases[terms,term[#,rate_]:>rate]] & /@ spcs];

(* Create list of term[] statements from parsing a rxn statement *) 
ProcessRxnToTerms[reaction:rxn[r_,p_,k_],t_Symbol]:=
Module[{spcs=SpeciesInRxnsys[{reaction}], rrate, spccoeffs,terms},
(* compute rate of this reaction *)
rrate = k (r/.{Times[b_,s_]:>s^b,Plus->Times});
(*for each species, get a net coefficient*)
spccoeffs=Coefficient[p-r,#]& /@ spcs;
(*create term for each species*)
terms=MapThread[term[#1,#2*rrate]&,{spcs, spccoeffs}];
(*change all species variables in the second arg in term[] to be functions of t*)
terms/.term[spc_,rate_]:>term[spc,rate/.s_/;MemberQ[spcs,s]:>s[t]]];


SimulateRxnsys[rxnsys_,endtime_,opts:OptionsPattern[NDSolve]]:=
 Module[{spcs=SpeciesInRxnsys[rxnsys],odesys=RxnsysToOdesys[rxnsys,Global`t]},
 Quiet[NDSolve[odesys, spcs, {Global`t,0,endtime},opts,MaxSteps->Infinity,AccuracyGoal->MachinePrecision],{NDSolve::"precw"}][[1]]]

(* FFNC Removing Unimolecular Reactions *)

(* tests if reaction is unimolecular *)

unimolReactionQ[rxn[r_, ___]] := 
 Not[Or[Head[r] === Times, Head[r] === Plus]]

(* convert a+b to {a,b} and 2a to {a,a} and a to {a} *)
listerize[ps_] := 
 Replace[Replace[
   ps, {ss_Plus :> List @@ ss, s_ :> {s}}], {_Integer -> Seq[], 
   c_Integer*s_ :> Seq @@ Table[s, {c}], s_ :> s}, {1}]

(* separate initial concentrations from the reactions 
returns {concentrations [as Association], reactions}  *)
separateConcs[
  crn_] :=
 {Merge[Cases[crn, conc[x_, c_] :> x -> c], Total],
  Cases[crn, rxn[___]]}

(* merges output of separateConcs into a single CRN *)
mergeConcs[concs_, rxns_] :=
  Join[rxns, KeyValueMap[conc[#1, #2] &, concs]]

(* splits reaction rxn[a,b+c,1] to {a,b+c} *)
splitReaction[rxn[x_, prods_, _]] := {x, prods}

(*
 The following is for ensuring that inputs are not the same species as the outputs 
 by not pushing the optimization all the way to the beginning.
 This is enabled by the keepInputLayer parameter in reduceFFNCCrn[].
 Inputs species can be identified by inputPrefix as the initial part of their name.
*)

(*
 Returns the species that are not input species based on the name prefix.
 We are potentially ok eliminating these as they are not inputs
*)
getNonInputSpecies[crn_, inputPrefix_]:=
  Complement[SpeciesInRxnsys[crn], SpeciesInRxnsysStringPattern[crn, inputPrefix<>"*"]]

(* Tests if reaction can be eliminated: must be unimolecular with the reactant in okSpecies *)
removableUnimolReactionQ[rxn[r_,___],okSpecies_]:=MemberQ[okSpecies,r]

reduceFFNCCrn[crn_, keepInputLayer_:False, inputPrefix_:""] :=
 Module[{concs, rxns, rpos, x, prods, c, nonInputSpecies},
  {concs, rxns} = separateConcs[crn];
  nonInputSpecies = getNonInputSpecies[crn, inputPrefix];
  
  While[True,
  
   (* find a unimolecular reaction we can eliminate*) 
   If[keepInputLayer, 
      rpos = FirstPosition[rxns,r:rxn[___]/;removableUnimolReactionQ[r,nonInputSpecies]],
      rpos = FirstPosition[rxns, rxn[___]?unimolReactionQ]
   ];
   (* if there still is a unimolecular reaction to eliminate, process it, otherwise we are done *)
   If [Not[MissingQ[rpos]],
    (*** process unimolecular reaction ***)
    {x, prods} = splitReaction[rxns[[First[rpos]]]];
    rxns = Delete[rxns, rpos];
    (* lookup x concentration; 0 if not found *)
    c = Lookup[concs, x, 0];
    KeyDropFrom[concs, x]; 
    (* add c to the initial concentrations of the products *)
    Scan[If[KeyExistsQ[concs, #], concs[#] += c, concs[#] = c] &, listerize[prods]];
    (* replace x with prods in all reactions *)
    rxns = Replace[rxns, x->prods, {2,4}]; 
    ,
    (* if no more reactions to eliminate, return new crn *)
    (* but first, expand all product expression: x+2(y+z) => x + 2y + 2z *)
    rxns = rxns /. rxn[r_, p_, k_] :> rxn[r, Expand[p], k];
    Return[mergeConcs[concs, rxns]]]]]


(* convert names like speciespos to species[pos] and speciesneg to species[neg] *)
signSymbolsInExpression[exp_, symbolList_] := exp /. Dispatch[(# -> signSymbol[#] & /@ symbolList)]

signSymbol[s_Symbol] :=
 Module[{name = SymbolName[s], sign, base},
  If[StringLength[name] <= 3, Return[s]];
  sign = StringTake[name, -3];
  If[sign == "pos",
   base = StringTake[name, StringLength[name] - 3];
   Return[Symbol[base][pos]]];
  If[sign == "neg",
   base = StringTake[name, StringLength[name] - 3];
   Return[Symbol[base][neg]]];
  Return[s]]

(* convert species names back from species[pos] to speciespos *) 
unsignSymbolsInExpression[exp_, symbolList_] := exp /. Dispatch[(# -> unsignSymbol[#] & /@ symbolList)]

unsignSymbol[e_[pos]] := Symbol[SymbolName[e] <> "pos"]
unsignSymbol[e_[neg]] := Symbol[SymbolName[e] <> "neg"]
unsignSymbol[e_] := e


(* splits rest of crn and signed concs *)
splitSignedConcs[crn_] :=
 {Select[crn, Not@*MatchQ[conc[_[pos | neg], _]]], 
  Select[crn, MatchQ[conc[_[pos | neg], _]]]}

(* gets the base part of base[pos] and base[neg] for all species *)
getSignedBases[crn_] := 
 DeleteDuplicates[Cases[SpeciesInRxnsys[crn], b_[_] :> b]]

(* Normalizes concentrations in the sense that {conc[speciespos, 1], conc[speciesneg, 2]} becomes {conc[speciesneg,1]} *)
(* Returns concs and Ifs of concs *)
normalizeConcs[crn_] :=
 Module[{signedconcs, restcrn},
  {restcrn, signedconcs} = splitSignedConcs[crn];
  Join[restcrn, normalizeSignedConcs[signedconcs]]]

normalizeSignedConcs[concs_] :=
 Replace[Normal[Merge[
    Replace[concs,
     {conc[b_[pos], v_] :> b -> v,
      conc[b_[neg], v_] :> b -> -v},
     {1}],
    Total]],
  (b_ -> v_) :> If[v >= 0, conc[b[pos], v], conc[b[neg], -v]],
  {1}]


(* Normalizes products of reactions in the sense that 2*speciespos + speciesneg becomes speciespos *)

normalizeProducts[crn_] :=
 Module[{bases, newcrn = crn, rules},
  bases = getSignedBases[crn];
  rules = Dispatch[#[neg] -> -#[pos] & /@ bases];
  newcrn = 
   Replace[newcrn, rxn[r_, p_, k_] :> rxn[r, p /. rules, k], {1}];
  newcrn = Replace[newcrn, rxn[r_, p_, k_] :> rxn[r, p /. {
        Times[c_?Negative, b_[neg]] :> -c b[pos],
        Times[c_?Negative, b_[pos]] :> -c b[neg]}, k], {1}];
  newcrn]


(* Normalizes whole CRN *)
normalizeCRN[crn_]:=
 Module[{crnsigned=signSymbolsInExpression[crn, SpeciesInRxnsys[crn]]},
  crnsigned=normalizeProducts[normalizeConcs[crnsigned]];
  unsignSymbolsInExpression[crnsigned, SpeciesInRxnsys[crnsigned]]]
  

(*****************************************)

(* Prints total and average number of products in CRN. *)
printProductsStats[rsys_] :=
Module[{products, totalCount, maxCount, currentCount, avgCount},
  products = Cases[rsys, rxn[_,x_,_]->x];
  totalCount = 0;
  maxCount = 0;
  For[i=1, i<=Length[products], i++,
    currentCount = Length[listerize[products[[i]]]];
    totalCount = totalCount + currentCount;
    If[currentCount > maxCount,
      maxCount = currentCount];
  ];
  Print["Total count of products in CRN: " <> ToString[totalCount]];
  avgCount = N[totalCount] / Length[products];
  Print["Avg count of products in CRN: " <> ToString[avgCount]];
  Print["Max count of products in a reaction: " <> ToString[maxCount]];
];
                                                                            
End[];
EndPackage[];
