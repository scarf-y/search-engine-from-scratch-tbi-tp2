class FSTDictionary:
    """
    Deterministic character-level finite-state transducer (FST) sederhana
    untuk pemetaan term string -> term_id.

    Implementasi ini fokus pada correctness dan kemudahan integrasi:
    - exact lookup term -> id
    - prefix traversal untuk suggestions
    """

    def __init__(self):
        # transitions[state][char] = next_state
        self.transitions = [{}]
        # final_outputs[state] = term_id
        self.final_outputs = {}

    def insert(self, term, output):
        state = 0
        for ch in term:
            next_state = self.transitions[state].get(ch)
            if next_state is None:
                next_state = len(self.transitions)
                self.transitions[state][ch] = next_state
                self.transitions.append({})
            state = next_state
        self.final_outputs[state] = output

    def lookup(self, term):
        state = 0
        for ch in term:
            next_state = self.transitions[state].get(ch)
            if next_state is None:
                return None
            state = next_state
        return self.final_outputs.get(state)

    def contains(self, term):
        return self.lookup(term) is not None

    def prefix_search(self, prefix, limit=10):
        """
        Mengembalikan list tuple (term, term_id) untuk term-term
        yang memiliki prefix tertentu.
        """
        state = 0
        for ch in prefix:
            next_state = self.transitions[state].get(ch)
            if next_state is None:
                return []
            state = next_state

        results = []
        stack = [(state, prefix)]
        while stack and len(results) < limit:
            current_state, current_term = stack.pop()
            if current_state in self.final_outputs:
                results.append((current_term, self.final_outputs[current_state]))

            # urutkan agar traversal stabil (lexicographic ascending)
            for ch, next_state in sorted(self.transitions[current_state].items(), reverse=True):
                stack.append((next_state, current_term + ch))
        return results

    @classmethod
    def from_id_to_str(cls, id_to_str):
        fst = cls()
        for term_id, term in enumerate(id_to_str):
            fst.insert(term, term_id)
        return fst
