import spacy

# # nlp = spacy.load('en_core_web_sm')
# def correct_sentence(doc, sentence):
#     # doc = nlp(sentence)
#     # Find the root of the sentence
#     root = [token for token in doc if token.head == token][0]

#     # If the root is a verb, return the original sentence
#     if root.pos_ == "VERB":
#         return sentence
#     # If the root is not a verb, find the verb and make it a main verb by adding "is" before it
#     else:
#         verbs = [token for token in doc if token.pos_ == "VERB"]
#         if verbs:
#             verb = verbs[0]
#             corrected_sentence = sentence.replace(str(verb), f"is {verb}")
#             return corrected_sentence
#     return sentence
# def extract_main_np(doc):
#     # Find the root of the sentence
#     root = [token for token in doc if token.head == token][0]
#     # Find the noun or noun phrase connected to the root
#     main_np = [child for child in root.children if child.pos_ in ["NOUN", "PROPN", "PRON"]]
    
#     # If a noun is found, return the noun phrase associated to it
#     if main_np:
#         noun = main_np[0]
#         return " ".join([str(token) for token in noun.subtree])
#     return doc
def extract_noun_phrase(text, doc, need_index=False):
    # text = text.lower()

    # doc = nlp(text)

    chunks = {}
    chunks_index = {}
    for chunk in doc.noun_chunks:
        for i in range(chunk.start, chunk.end):
            chunks[i] = chunk
            chunks_index[i] = (chunk.start, chunk.end)

    for token in doc:
        if token.head.i == token.i:
            head = token.head

    if head.i not in chunks:
        children = list(head.children)
        if children and children[0].i in chunks:
            head = children[0]
        else:
            if need_index:
                return text, [], text
            else:
                return text

    head_noun = head.text
    head_index = chunks_index[head.i]
    head_index = [i for i in range(head_index[0], head_index[1])]

    sentence_index = [i for i in range(len(doc))]
    not_phrase_index = []
    for i in sentence_index:
        not_phrase_index.append(i) if i not in head_index else None

    head = chunks[head.i]
    if need_index:
        return head.text, not_phrase_index, head_noun
    else:
        return head.text