import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
T=[("THE ONE-PAGE RESULT TEST",14,"bold"),("",6,"normal"),
("QUESTION: can a reader be convinced by the figures alone?",10,"bold"),("",5,"normal"),
("Fig 1  Same policy + same scenario + same policy noise -> different outcomes.",9,"normal"),
("        Physics realization alone decides success.",9,"normal"),
("Fig 2  Outcomes bifurcate (0.018 m vs 0.304 m; threshold 0.04 m), they do not jitter.",9,"normal"),
("        Sensitive scenarios show 2.7x the contact-timing variability (association only).",9,"normal"),
("Fig 3  14.8% of single-realization views flip the SIGN vs an R=3 reference;",9,"normal"),
("        25.0% flip the practical category. Range across contrasts: 0%-56%.",9,"normal"),
("Fig 4  Resolution floor: ~7.6-8.1 pp at R=1; ~4.4-4.7 pp at R=3.",9,"normal"),
("Fig 5  Calibration predicts SIGN reliability (67%->97%) but NOT category (flat 72-78%).",9,"normal"),
("",7,"normal"),
("VERDICT",11,"bold"),
("The measurement contribution stands on frozen data alone:",9,"normal"),
("  a demonstrated failure mode, a quantified floor, a calibration that half-works.",9,"normal"),
("The honest headline is a RESOLUTION LIMIT, not a claim about fast samplers.",9,"bold"),
("",7,"normal"),
("WHAT THE FIGURES DO NOT SHOW",11,"bold"),
("  - No contrast is significant: 0 of 12 CIs exclude zero.",9,"normal"),
("  - No equivalence claim is licensed; absence of resolution is not evidence of parity.",9,"normal"),
("  - Slot/environment-index effects: NEED-E1.",9,"normal"),
("  - Causality of contact timing: MISSING-EVIDENCE.",9,"normal"),
("",7,"normal"),
("CORRECTIONS APPLIED IN THIS PACKAGE",11,"bold"),
("  - Headline is 14.8%: strict sign REVERSALS. Supersedes 24% and 21.3%.",9,"normal"),
("  - 7 of 108 views are exact ties (delta=0); a tie is not a reversal.",9,"normal"),
("  - Same tie bug inflated the Fig 5 low bin: 44.4% -> corrected 66.7%.",9,"normal"),
("  - Pooled == unweighted (0.0 pp): weighting is not driving the number.",9,"normal"),
("  - The 5 pp band is a PREDECLARED PROJECT CONVENTION, not a community standard.",9,"normal"),
("  - The 108 views are 12 contrasts x 9 NESTED views, never 108 experiments.",9,"normal")]
fig=plt.figure(figsize=(8.5,11)); y=0.96
for t,s,wt in T:
    fig.text(0.06,y,t,fontsize=s,fontweight=wt,va="top",family="DejaVu Sans"); y-=0.0165+s*0.0016
with PdfPages("experiments/topconf/paper_figures/ONE_PAGE_RESULT_TEST.pdf") as pp: pp.savefig(fig)
plt.close(fig); print("wrote ONE_PAGE_RESULT_TEST.pdf")
