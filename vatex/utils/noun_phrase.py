import spacy

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