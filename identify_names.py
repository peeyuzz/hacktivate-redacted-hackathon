import spacy
import io

nlp = spacy.load("en_core_web_sm")

def identify_names(text_boxes):
    names = []
    for item in text_boxes:
        doc = nlp(item['text'])
        for ent in doc.ents:
            print(ent.text + '-' + str(ent.start_char) + '-' + str(ent.end_char) + '-' + ent.label_ + '-' + str(spacy.explain(ent.label_)))
            if ent.label == "PERSON":
                names.append({
                    'name': ent.text,
                    'box': item['box'],
                    'page': item['page']

                })

    return names