# Author - Samantha Mahendran for RelEx_Colocation
"""
Fixes the original span issues in the ann files
"""
def find_alternative_span(start, end, span, doc, left_boundary=10, right_boundary = 20):

    span_left = None
    span_right = None

    #if the given span doesn't capture the correct span
    if span is None:
        word_left = 1

        # expand the span on the left to try find the correct span
        while span_left is None:
            s1 = start - word_left
            span_left = doc.char_span(s1, end)
            if span_left:
                new_span = span_left

            #if the correct span is not found until the given boundary stop expanding
            if word_left == left_boundary:
                if span_left is None:
                    word_right = 1

                    # expand the span on the right to try find the correct span
                    while span_right is None:
                        e1 = end + word_right
                        span_right = doc.char_span(start, e1)
                        if span_right:
                            new_span = span_right

                        # if the correct span is not found until the given boundary stop expanding
                        if word_right == right_boundary:
                            if span_right is None:
                                new_span = None
                            break
                        word_right = word_right + 1

                break
            word_left = word_left + 1
    else:
        # if the given span matches the correct span return the given span
        new_span = span

    return new_span