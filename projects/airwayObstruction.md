# Transforming Airway Obstruction Diagnosis with AI-Powered 3D Shape Analysis

**Authors:** Lucie Dole, Claudia T. Mattos, Jonas Bianchi, Heesoo Oh, Karine Evangelista, José Valladares Neto, Sergio L. Mota-Júnior, Lucia Cevidanes, Juan Carlos Prieto

---

[![Airway Obstruction Model Explainability](https://img.youtube.com/vi/Ek2daxgTEgE/0.jpg)](https://youtu.be/Ek2daxgTEgE)




## A New Frontier in Diagnosing Pediatric Airway Obstruction

Enlarged adenoids are a common cause of airway obstruction in children and adolescents, leading to serious consequences like obstructive sleep apnea (OSA), impaired cognitive function, and cardiovascular risks. Diagnosing these obstructions early is vital—yet the gold-standard methods, such as polysomnography, are costly, time-consuming, and impractical for large-scale screening.

Our research introduces a breakthrough: a deep learning-based tool that automatically analyzes 3D scans of the airway to detect obstruction and determine its severity. Using cone-beam computed tomography (CBCT), this tool brings speed, accuracy, and explainability to the diagnostic process.

---

## Why Traditional Methods Fall Short

While methods like Home Sleep Apnea Testing (HSAT) and clinical questionnaires offer some convenience, they often lack sensitivity and specificity. Recent advancements in artificial intelligence have improved single-modality analysis of physiological signals (like ECG and oxygen saturation), but none have tackled the airway obstruction challenge with a fully 3D, explainable model—until now.

---

## Enter SaxiMHAFB: A Multi-Modal Deep Learning Model

Our proposed solution, **SaxiMHAFB**, combines two 3D representations of airway anatomy:

- **Multi-view projections**: capture 2D images of the airway from multiple angles.
- **Point-cloud sampling**: preserves fine-grained geometric features by analyzing a random set of surface points.

Together, these approaches capture both global and local morphology, providing a rich, interpretable picture of airway structure. The model outputs either a severity classification (from Grade 1 to Grade 4) or a predicted nasopharynx airway obstruction (NAO) ratio.

---

## Real Results from Real Patients

Using over 400 anonymized CBCT scans from multiple international clinical centers, we trained our model to identify obstruction with impressive results:

- **Binary classification** (presence or absence of obstruction): **81.88% accuracy**
- **Severity classification** (Grades 1–4): **55.94% accuracy**, with strong performance in identifying severe cases
- **Regression of obstruction ratio (NAO)**: Mean Absolute Error (MAE) of **10.17%**

These results demonstrate not just accuracy, but **clinical relevance**—particularly in detecting urgent, high-severity cases.

---

## Explainability Built In

AI in medicine must be interpretable. That’s why we integrated **Grad-CAM** heatmaps into the multi-view component of our model. These visualizations show exactly which regions influenced the model’s predictions—usually the upper-mid airway, consistent with clinical insight.

> Note: Current explainability features apply only to the multi-view branch; future updates will include point-cloud visualizations for full transparency.

---

## Why This Matters

Early and accurate detection of airway obstruction can significantly improve outcomes for children. Our model offers a fast, non-invasive, and interpretable solution that clinicians can use as a **screening tool or decision support system**. By bridging the gap between cutting-edge AI and practical clinical needs, this research paves the way for smarter, scalable diagnostics.

---

## What’s Next?

To further enhance performance, especially in distinguishing subtle, mild-to-moderate obstructions, our future efforts will:

- Incorporate patient-specific metadata (e.g., age, symptoms)
- Extend explainability to the point-cloud branch
- Increase dataset diversity to boost generalizability
- Explore integration with functional imaging (e.g., airflow simulations)

---

## Final Thoughts

SaxiMHAFB isn’t just another deep learning model. It’s a **step toward transforming pediatric airway care**—making diagnostics faster, more reliable, and easier to interpret. With continued development, it could become an indispensable tool for otolaryngologists, orthodontists, and sleep specialists worldwide.

---

**🧪 Research supported by:** NIH Grant R01-DE024450  
**📂 Code & Model Access:** [ShapeAXI on GitHub](https://github.com/DCBIA-OrthoLab/ShapeAXI)  
**📋 IRB Approval:** University of Michigan (HUM00251245)

---
