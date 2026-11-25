# تقريرك: التفسير والمعايرة — Tamweel Lite

**الحالة:** جاهز للمراجعة؛ لا يعني اعتمادًا أو درجة

**مصدر التفسير:** LIVE. **النموذج والمعايرة:** LIVE. **السعة:** CAPACITY_REVIEW_REQUIRED.

## النموذج والأدوار
LightGBM موزون، 80 شجرة. الهدف حدث تعثر اصطناعي خلال90 يومًا بعد الطلب. الأدوار منفصلة زمنيًا وبالعملاء: تدريب 2516، معايرة 584 (40 موجب)، سياسة 589، تقييم 1733. الفجوات والتداخلات مستبعدة. سبق استخدام بيانات التقييم في الدورة، فهي ليست اختبارًا نهائيًا لم يمسّ.

## التفسير العام والمحلي
Permutation يقيس انخفاضAP على التقييم؛ إشارات المنطقة تُبدّل معًا. SHAP يفسر النموذج الخام بوحدةlog-odds وخلفية مسارات أشجار التدريب. base+sum(SHAP)=raw margin، ثمsigmoid للمجموع فقط. القيم ليست نقاط احتمال ولا تفسيرًا مباشرًا للنموذج المعاير.

Global explanations, such as permutation importance and the SHAP beeswarm plot, describe overall model behavior, feature importance rankings, and directional trends across the evaluation sample. Both methods identified `bureau_score` and `dti` as the two most influential features, with permutation AP drops of 0.1270 and 0.0690, respectively, and mean absolute SHAP values of 0.9042 and 0.5437 log-odds.

In contrast, local explanations, such as the SHAP waterfall plot and reason codes, break down individual feature contributions for a specific loan application. For example, application TR-009585 had a raw score of 0.9031 and a calibrated probability of 0.4795, with `bureau_score`, `dti`, and `loan_amount_sar` being its three largest positive contributors.

SHAP values are measured in log-odds rather than probability points. The raw margin is calculated as the base value plus the sum of all SHAP contributions, and the raw probability is obtained by applying the sigmoid function to the final aggregated margin.

The background distribution is based on training tree paths, and feature inputs reflect values after fit-only imputation. The maximum observed additivity error was approximately 5.70 × 10⁻¹⁵, confirming that the SHAP contributions accurately reconstructed the model's raw margins within numerical precision.

الطلب الاصطناعي TR-009585: الدرجة الخام 0.90308 والاحتمال المعاير 0.47952. اختير أعلى درجة داخل عينةSHAP دون استخدام النتيجة الفعلية.
- استخدم النموذج درجة ائتمانية اصطناعية عند الطلب بالقيمة 497 لرفع درجته الخام؛ لا يثبت ذلك السببية. (SHAP=+2.2693 log-odds؛ قيمة معوضة: False)
- استخدم النموذج نسبة الالتزام مع القسط المقترح إلى الدخل بالقيمة 1.2806 لرفع درجته الخام؛ لا يثبت ذلك السببية. (SHAP=+1.1981 log-odds؛ قيمة معوضة: False)
- استخدم النموذج مبلغ التمويل المطلوب بالقيمة 93437.3 لرفع درجته الخام؛ لا يثبت ذلك السببية. (SHAP=+0.2007 log-odds؛ قيمة معوضة: False)

Reason statements are derived from at most three leading positive SHAP contributions and describe model behavior rather than causality, fairness, or protected-trait effects. A positive contribution does not necessarily mean that a feature's value is high, and changing a feature does not guarantee a change in the prediction or actual default outcome. Correlated features may share predictive information, affecting their individual importance measurements.

For application TR-009585, changing the bureau score from 497 to either 496 or 498 resulted in an unchanged raw probability of 0.903079 and calibrated probability of 0.479518 at the reported precision. All three leading positive reasons remained unchanged. This demonstrates local stability under the tested perturbations, but it does not establish general model robustness or causality.

## دليل المعايرة
على 1733 صفًا و139 موجب: Brier 0.113027 → 0.067112؛ ECE 0.146871 → 0.022486. عشر حاويات متساوية العرض مع أعدادها فيday4_reliability_bins.csv. AP 0.258677 → 0.258677؛ ROC-AUC 0.770804 → 0.770804. هذه نتائج هذه العينة وليست ضمانًا لتحسن مستقبلي.

The sigmoid calibration mapping was fitted exclusively on 584 applications from July to September 2023 and evaluated on a separate, unseen period from July to December 2024, containing 1,733 applications.

Following calibration, the Brier score decreased from 0.11303 to 0.06711, ECE decreased from 0.14687 to 0.02249, and log loss decreased from 0.35799 to 0.24675. Meanwhile, ROC-AUC and Average Precision remained unchanged at 0.77080 and 0.25868, respectively.

These results demonstrate that sigmoid calibration improved probability reliability on the held-out evaluation period without changing the model's ranking ability. However, this improvement does not guarantee reliable calibration in future periods.

## الاستقرار
200 تكرارbootstrap صالح بسحب العملاء؛ فترات مئينية95% مع تثبيت النموذج والمعاير. لا تشمل تعلم النموذج أو المعايرة أو الانجراف المستقبلي، ولا تصف احتمال فرد. انحرافAP بين ربعي التقييم وصفي فقط. اختبارbureau_score±1 نُفذ؛ راجع day4_local_stability.csv.

The paired customer-cluster bootstrap captures sampling variability across customer-level clusters under a fixed model and calibrator.

The resulting 95% percentile interval for Average Precision was [0.19744, 0.33785] for both raw and calibrated predictions. The interval for the Brier score change was [-0.05418, -0.03741]. Since the entire Brier change interval is below zero, the results support an improvement in probability calibration under the evaluated conditions.

However, these intervals exclude uncertainty associated with model retraining, calibration parameter refitting, and future macroeconomic or temporal drift.

## العتبة ومنطقة المراجعة
العتبة الخام 0.5881953696965011 اختيرت علىpolicy بخسارة10×FN+FP وسقف12% ثم نُقلت إلى 0.17331013263107387. لم تعدل باستخدام التقييم. المنطقة[0.15331, 0.19331] تشخيصية بعرض±0.02 وليست فترة ثقة. الاتحاد يحسب الطلب مرة واحدة.
- 2024Q3: السقف 100، الإشارات 97، اتحاد المراجعة 109.
- 2024Q4: السقف 107، الإشارات 109، اتحاد المراجعة 122.

The ±0.02 probability review zone serves as a diagnostic band for identifying near-threshold applications. These applications are combined with core risk flags as a single union set to assess total review volume against the 12% per-period capacity limit.

The raw policy threshold of 0.588195 was transported to a calibrated threshold of 0.17331. In 2024 Q3, the combined review candidates reached 109 against a capacity of 100. In 2024 Q4, they reached 122 against a capacity of 107, with the 109 core risk flags alone already exceeding capacity.

Consequently, the notebook returned CAPACITY_REVIEW_REQUIRED. The policy should not be retuned using the evaluation data; any adjustments must be developed and assessed through a new, independent validation process.

عند تجاوز السعة، وثّق الحاجة إلى تصميم سياسة جديدة على بيانات تطوير وتقييمها بدليل جديد. لا ترفع السقف ولا تقص الحالات بعد رؤية النتيجة. التفسير ليس سببية أو شهادة عدالة، والخسارة وحدات تعليمية لا رسوم أو خصم درجات. لا يستخدم هذا التمرين لتمويل حقيقي.
