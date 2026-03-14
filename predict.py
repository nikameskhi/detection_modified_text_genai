import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "results_roberta_pan24"
MAX_LENGTH = 128  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()


@torch.inference_mode()
def predict(text: str):
    
    tokens = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    logits = model(**tokens).logits
    probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    # Prediction of class
    pred = int(probs.argmax())
    label = "Human" if pred == 0 else "Modified/Machine"

    return label, float(probs[0]), float(probs[1])


if __name__ == "__main__":
    # modified
    
    # sample_text = (
    #     "Pandemic's Lasting Impact: Millions of Jobs Unlikely to Return, Spurring Career Changes and Retraining As the world grapples with the ongoing COVID-19 pandemic, a harsh reality is setting in: millions of jobs affected by the crisis are unlikely to return, necessitating career changes and retraining for the unemployed. According to a recent report by the McKinsey Global Institute, up to 20% of the workforce in some countries may need to switch occupational categories or industries to adapt to the new economic landscape.Federal Reserve Chair Jerome H. Powell acknowledged the transition to a different economy in a recent speech, stating, ""The pandemic has accelerated trends that were already in motion, such as the growth of e-commerce and the increasing use of automation in many industries.""A significant portion of the unemployed are considering changing their occupations or fields of work. In a recent survey by Glassdoor, 60% of respondents said they were open to exploring new career opportunities. However, a lack of financial resources for retraining poses a challenge for many unemployed people.Automation is accelerating in industries such as retail and manufacturing, potentially reducing the need for human labor. Online pet supply company Chewy, for example, recently opened a fully automated fulfillment center in Florida, which requires far fewer workers than traditional brick-and-mortar stores. Job postings indicate a decline in demand for various occupations, suggesting a shift in the job market.Despite some sectors experiencing growth, the unpredictability of which industries will thrive makes job transitions difficult. Initiatives like Michigan's 'Futures for Frontliners' program, which offers free tuition for retraining to front-line workers, are gaining traction.Bill Gates, founder-turned-philanthropist of Microsoft, noted, ""The pandemic has accelerated the shift towards automation and remote work, which will continue to reshape the job market in the coming years.""Susan Lund, Head of the McKinsey Global Institute, added, ""The pandemic has created a unique opportunity for workers to upskill and reskill, but it will require significant investment in education and training to ensure that everyone can adapt to the changing job market.""Brad Hershbein, Senior Economist at the W.E. Upjohn Institute for Employment Research, observed, ""The pandemic has accelerated trends that were already in motion, such as the growth of e-commerce and the increasing use of automation in many industries.""David Autor, Economist at Massachusetts Institute of Technology, stated, ""The pandemic has exposed the vulnerabilities of certain industries and occupations, but it has also created new opportunities for workers to adapt and thrive in a rapidly changing job market.""Sumit Singh, CEO of Chewy, said, ""The pandemic has accelerated the shift towards online shopping and automation, which has created new opportunities for companies like Chewy to innovate and grow.""Diane Pelkey, spokeswoman for Chewy, noted, ""The pandemic has forced companies to rethink the way they do business, and Chewy is no exception. Our fully automated fulfillment center is just one example of how we are adapting to the new normal.""Stephanie Wissink, Managing Director at Jefferies, observed, ""The pandemic has created a challenging environment for workers and companies alike, but it has also accelerated the shift towards automation and remote work, which will continue to reshape the job market in the coming years.""Andrew Chamberlain, Chief Economist at Glassdoor, stated, ""The pandemic has exposed the vulnerabilities of certain industries and occupations, but it has also created new opportunities for workers to adapt and thrive in a rapidly changing job market.""In conclusion, the COVID-19 pandemic has accelerated trends that were already in motion, such as the growth of e-commerce and the increasing use of automation in many industries. Millions of jobs affected by the crisis are unlikely to return, necessitating career changes and retraining for the unemployed. While some sectors are experiencing growth, the unpredictability of which industries will thrive makes job transitions difficult. Initiatives like Michigan's 'Futures for Frontliners' program are gaining traction, offering free tuition for retraining to front-line workers. The pandemic has exposed the vulnerabilities of certain industries and occupations, but it has also created new opportunities for workers to adapt and thrive in a rapidly changing job market.")
    
    # human
    sample_text = "'Rust' assistant director who handed Alec Baldwin prop gun subpoenaed after declining investigation interviewSANTA FE, N.M. — The assistant director who handed Alec Baldwin a prop gun that killed a cinematographer on a New Mexico film set must make himself available for an interview with state workplace safety regulators, a judge has decided.District Judge Bryan Biedscheid on Friday granted a request by the Occupational Health and Safety Bureau of the state Environment Department to issue a subpoena to Dave Halls, assistant director for the movie ""Rust,"" local news outlets reported.Cinematographer Halyna Hutchins was killed and director Joel Souza was wounded in the Oct. 21 shooting on the Bonanza Creek Ranch film set near Santa Fe.Safety officials tried twice since Nov. 2 to interview Halls for their investigation but he declined both times through his attorney and said he wouldn't agree to an interview until a criminal investigation into the shooting is complete, a compliance officer wrote Wednesday in an affidavit in support of the subpoena request.The interview with Halls is needed because he had responsibilities for set safety, knew who was present during the shooting and had handled the gun, the application said.Rebecca Roose, deputy cabinet secretary of the Environment Department, told the Santa Fe New Mexican that the department proposed a Tuesday interview but that the judge could set another date or Halls' attorney could fight the subpoena.Halls' attorney, Lisa Torracco, on Saturday did not immediately respond to a voicemail left by The Associated Press seeking comment.However, KOB-TV reported that Torraco told the station that Halls will cooperate with state investigators."
    label, p_human, p_machine = predict(sample_text)
    print("Prediction:", label)
    print("P(Human):", p_human)
    print("P(Machine):", p_machine)