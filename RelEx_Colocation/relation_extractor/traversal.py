#Authour - Samantha Mahendran for RelEx_Colocation

from RelEx_Colocation.utils import alternative_span

# traverse only the right side of the drug mention
def traverse_right_only(sorted_entities, result, f):
    len_SE = len(sorted_entities)

    for id, start, end, label, mention in sorted_entities:
        index = sorted_entities.index((id, start, end, label, mention))

        if label == 'Drug':
            return result

        ind_right = index + 1
        count = 1
        while ind_right < len_SE:
            sorted_id, sorted_start, sorted_end, sorted_label, sorted_mention  = sorted_entities[ind_right]
            if sorted_label == 'Drug':
                result['relations'].append((label, sorted_label, id, sorted_id, f))
                break
            ind_right = ind_right + count

    return result

# traverse only the left side of the drug mention
def traverse_left_only(sorted_entities, result, f):

    for id, start, end, label, mention in sorted_entities:
        index = sorted_entities.index((id, start, end, label, mention))

        if label == 'Drug':
            return result

        ind_left = index - 1
        count = 1
        while ind_left > -1:
            sorted_id, sorted_start, sorted_end, sorted_label, sorted_mention  = sorted_entities[ind_left]
            if sorted_label == 'Drug':
                result['relations'].append((label, sorted_label, id, sorted_id, f))
                break
            ind_left = ind_left - count

    return result

#traverse right first then left of the drug mention
def traverse_right_left(sorted_entities, result, f):
    len_SE = len(sorted_entities)

    for id, start, end, label , mention in sorted_entities:
        index = sorted_entities.index((id, start, end, label, mention))

        if label == 'Drug':
            return result

        ind_left = index - 1
        ind_right = index + 1
        count = 1

        while ind_left > -1 and ind_right < len_SE:
            right_id, right_start, right_end, right_label, right_mention = sorted_entities[ind_right]
            if right_label == 'Drug':
                result['relations'].append((label, right_label, id, right_id, f))
                break
            ind_right = ind_right + count

            left_id, left_start, left_end, left_label, left_mention = sorted_entities[ind_left]
            if left_label == 'Drug':
                result['relations'].append((label, left_label, id, left_id, f))
                break
            ind_left = ind_left - count

    return result

#traverse left first then right of the drug mention
def traverse_left_right(sorted_entities, result, f):
    len_SE = len(sorted_entities)

    for id, start, end, label, mention in sorted_entities:
        index = sorted_entities.index((id, start, end, label, mention))

        if label == 'Drug':
            return result

        ind_left = index - 1
        ind_right = index + 1
        count = 1

        while ind_left > -1 and ind_right < len_SE:
            left_id, left_start, left_end, left_label, left_mention = sorted_entities[ind_left]
            if left_label == 'Drug':
                result['relations'].append((label, left_label, id, left_id, f))
                break
            ind_left = ind_left - count

            right_id, right_start, right_end, right_label, right_mention = sorted_entities[ind_right]
            if right_label == 'Drug':
                result['relations'].append((label, right_label, id, right_id, f))
                break
            ind_right = ind_right + count

    return result

#traverse both sides within the sentence boundary
def traverse_within_sentence(sorted_entities, result, f,doc):
    len_SE = len(sorted_entities)

    for id, start, end, label, mention in sorted_entities:
        index = sorted_entities.index((id, start, end, label, mention))

        if label == 'Drug':
            return result

        span = doc.char_span(start, end)
        #finds the correct span if the given span doesn't match the correct span
        new_span = alternative_span.find_alternative_span(start, end, span, doc)

        if new_span:
            span_sent = str(new_span.sent).split()
            ind_left = index - 1
            count_left = 1

            while ind_left > -1:
                left_id, left_start, left_end, left_label, left_mention = sorted_entities[ind_left]
                span_left = doc.char_span(left_start, left_end)
                new_span_left = alternative_span.find_alternative_span(start, end, span_left, doc)

                if left_label == 'Drug':
                    if any(str(new_span_left[0]) in s for s in span_sent):
                        result['relations'].append((label, left_label, id, left_id, f))
                    else:
                        break
                ind_left = ind_left - count_left

            ind_right = index + 1
            count_right = 1

            while ind_right < len_SE:
                right_id, right_start, right_end, right_label, right_mention = sorted_entities[ind_right]
                span_right = doc.char_span(right_start, right_end)
                new_span_right = alternative_span.find_alternative_span(start, end, span_right, doc)

                if right_label == 'Drug':
                    if any(str(new_span_right[0]) in s for s in span_sent):
                        result['relations'].append((label, right_label, id, right_id, f))
                    else:
                        break

                ind_right = ind_right + count_right

    return result


