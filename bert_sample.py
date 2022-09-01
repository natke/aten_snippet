

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from torch_ort import ORTInferenceModule, OpenVINOProviderOptions

provider_options = OpenVINOProviderOptions(backend = "GPU", precision = "FP16")

tokenizer = AutoTokenizer.from_pretrained(
            "textattack/bert-base-uncased-CoLA")
model = AutoModelForSequenceClassification.from_pretrained(
        "textattack/bert-base-uncased-CoLA")
# Wrap model in ORTInferenceModule to prepare the model for inference using OpenVINO Execution Provider on CPU
model = ORTInferenceModule(model)
text = "Replace me any text by you'd like ."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
# Post processing
logits = output.logits
logits = logits.detach().cpu().numpy()
# predictions
pred = np.argmax(logits, axis=1).flatten()
print("Grammar correctness label (0=unacceptable, 1=acceptable)")
print(pred)

