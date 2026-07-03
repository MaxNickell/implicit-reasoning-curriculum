import random
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Set, Tuple

import numpy as np

try:
    import generator
except ImportError:
    raise ImportError(
        "C++ generator module not found. Rebuild with:\n"
        "  pip install -e ."
    )

from faker import Faker


PREDICATE_TEMPLATES = [
    "If {name} is {adj_a}, then {name} is {adj_b}.",
    "If {name} is {adj_a}, then they are {adj_b}.",
    "If a person is {adj_a}, they are {adj_b}.",
    "Everyone that is {adj_a} is {adj_b}.",
    "If someone is {adj_a}, then they are {adj_b}."
]


@dataclass
class NLExample:
    input_text: str
    output_texts: List[str]
    labels: List[Any]
    output_vector: Any = None
    metadata: Dict = None


class NameAttributeGenerator:

    def __init__(self, seed=None):
        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)
            random.seed(seed)
        self.fake.unique.clear()

    def generate_names(self, n: int) -> List[str]:
        self.fake.unique.clear()
        return [self.fake.unique.first_name() for _ in range(n)]

    def random_syllable(self) -> str:
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        patterns = ["CV", "VC", "CVC", "CVV", "CCV", "VCV", "VCC"]
        pattern = random.choice(patterns)

        syllable = ""
        for char in pattern:
            if char == "C":
                syllable += random.choice(consonants)
            elif char == "V":
                syllable += random.choice(vowels)
        return syllable

    def generate_attribute(self) -> str:
        return self.random_syllable() + self.random_syllable()

    def generate_attributes(self, n: int) -> List[str]:
        attributes = set()
        while len(attributes) < n:
            attributes.add(self.generate_attribute())
        attributes = sorted(list(attributes))
        random.shuffle(attributes)
        return attributes


def _convert_to_int(value):
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    return int(value)


def generate_predicate_lines(edges, id_to_attr, name):
    lines = ["Suppose we have the following facts:"]

    for (A, B) in edges:
        adj_A = id_to_attr[A]
        adj_B = id_to_attr[B]

        template = random.choice(PREDICATE_TEMPLATES)
        line = template.format(name=name, adj_a=adj_A, adj_b=adj_B)
        lines.append(line)

    return lines


class NaturalLanguageGraphGenerator:

    def __init__(self, max_input_size: int, seed: int = None, debug: bool = False):
        self.max_input_size = max_input_size
        self.seed = seed
        self.debug = debug
        if seed is not None:
            generator.set_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        # Token definitions (must match C++ generator's encoding)
        self.QUERY_PREFIX_TOKEN = (max_input_size - 5) // 3 + 4
        self.PADDING_TOKEN = (max_input_size - 5) // 3 + 3
        self.EDGE_PREFIX_TOKEN = (max_input_size - 5) // 3 + 2
        self.PATH_PREFIX_TOKEN = (max_input_size - 5) // 3 + 1

        self.MAX_NODE_ID = (max_input_size - 5) // 3 - 1

        self.name_gen = NameAttributeGenerator(seed)

    def _parse_symbolic_output(self, inputs: np.ndarray, outputs: np.ndarray,
                               labels: np.ndarray = None) -> List[Dict]:
        examples = []

        for i in range(inputs.shape[0]):
            input_seq = [x for x in inputs[i] if x != self.PADDING_TOKEN]

            edges = []
            query = None
            path = []

            j = 0
            while j < len(input_seq):
                if input_seq[j] == self.EDGE_PREFIX_TOKEN:
                    edges.append((_convert_to_int(input_seq[j + 1]),
                                  _convert_to_int(input_seq[j + 2])))
                    j += 3
                elif input_seq[j] == self.QUERY_PREFIX_TOKEN:
                    query = (_convert_to_int(input_seq[j + 1]),
                             _convert_to_int(input_seq[j + 2]))
                    j += 3
                elif input_seq[j] == self.PATH_PREFIX_TOKEN:
                    j += 1
                    path = [_convert_to_int(x) for x in input_seq[j:]]
                    break
                else:
                    j += 1

            # Handle multiple outputs
            if outputs.ndim == 1:
                output_vals = [_convert_to_int(outputs[i])]
                output_vector = None
            else:
                output_row = outputs[i]
                output_vector = output_row.copy()

                if output_row.dtype == np.float32 or output_row.dtype == np.float64:
                    output_vals = [_convert_to_int(idx) for idx, val in enumerate(output_row) if val > 0.5]
                else:
                    output_vals = [_convert_to_int(idx) for idx, val in enumerate(output_row) if val == 1]

                if len(output_vals) == 0:
                    output_vals = [_convert_to_int(output_row.argmax())]

            example = {
                'edges': edges,
                'query': query,
                'path': path,
                'outputs': output_vals,
                'output_vector': output_vector,
                'label': _convert_to_int(labels[i]) if labels is not None else None
            }

            if self.debug:
                if output_vector is not None and sum(output_vector) > 1:
                    print(f"\nParsed example {i}:")
                    print(f"  Edges: {edges}")
                    print(f"  Query: {query}")
                    print(f"  Path: {path}")
                    print(f"  Outputs: {output_vals}")
                    print(f"  Output vector: {output_vector}")

            examples.append(example)

        return examples

    def _generate_search_nl(self, graph_data: Dict,
                            use_different_names: bool = False) -> NLExample:
        edges = graph_data['edges']
        query = graph_data['query']
        path = graph_data['path']
        outputs = graph_data['outputs']
        output_vector = graph_data.get('output_vector')
        label_id = graph_data.get('label')

        # Build the actual node set from the sample
        node_ids: Set[int] = set()
        for a, b in edges:
            node_ids.add(int(a))
            node_ids.add(int(b))
        if query:
            node_ids.add(int(query[0]))
            node_ids.add(int(query[1]))
        for o in outputs:
            node_ids.add(int(o))

        # Membership-based path filtering
        path = [int(n) for n in path]
        filtered_path = [n for n in path if n in node_ids]

        if self.debug and len(filtered_path) != len(path):
            dropped = [n for n in path if n not in node_ids]
            print(f"[SEARCH][WARN] Dropped {len(dropped)} path node(s) "
                  f"not present in graph node set: {dropped}")

        # Prepare names/attributes
        num_nodes = len(node_ids)
        all_names = self.name_gen.generate_names(num_nodes if use_different_names else 1)
        all_attributes = self.name_gen.generate_attributes(num_nodes)
        sorted_nodes = sorted(node_ids)
        id_to_attr = {node_id: all_attributes[i] for i, node_id in enumerate(sorted_nodes)}
        name = all_names[0]

        # Facts in NL
        lines = generate_predicate_lines(edges, id_to_attr, name)

        # Build the question text
        X, Y = query
        start_attr = id_to_attr[X]
        goal_attr = id_to_attr[Y]

        if filtered_path:
            path_text = f"{name} is {id_to_attr[filtered_path[0]]}"
            for node_id in filtered_path[1:]:
                path_text += f". {name} is {id_to_attr[node_id]}"
            question = (f"Given that {name} is {start_attr}, and we want to prove "
                        f"{name} is {goal_attr}.\n\nProof: {path_text}. "
                        f"{name} is")
        else:
            question = (f"Given that {name} is {start_attr}, and we want to prove "
                        f"{name} is {goal_attr}.\n\n"
                        f"Proof: {name} is")

        # Sanity checks (debug only)
        current_node = filtered_path[-1] if filtered_path else X
        one_hop_neighbors: Set[int] = {b for (a, b) in edges if a == current_node}

        if self.debug:
            if label_id is not None and label_id not in one_hop_neighbors:
                print(f"[SEARCH][WARN] Label {label_id} is not a one-hop neighbor of "
                      f"current node {current_node}. Neighbors: {sorted(one_hop_neighbors)}")

            invalid_outputs = [o for o in outputs if o not in one_hop_neighbors]
            if invalid_outputs:
                print(f"[SEARCH][WARN] Outputs not one-hop from current node {current_node}: "
                      f"{invalid_outputs}. Neighbors: {sorted(one_hop_neighbors)}")

        # Map outputs to attribute strings
        answer_attrs = [id_to_attr[o] for o in outputs]

        # Compose final NL input
        facts = " ".join(lines)
        input_text = facts + " " + question

        return NLExample(
            input_text=input_text,
            output_texts=answer_attrs,
            labels=outputs,
            output_vector=output_vector,
            metadata={'node_mapping': id_to_attr}
        )

    def generate_batch(self, batch_size: int, **kwargs) -> List[NLExample]:
        max_lookahead = kwargs.get('max_lookahead', 5)
        max_edges = kwargs.get('max_edges', (self.max_input_size - 5) // 3)
        reserved_inputs = kwargs.get('reserved_inputs', set())
        distance_from_start = kwargs.get('distance_from_start', -1)
        max_prefix_vertices = kwargs.get('max_prefix_vertices', self.max_input_size)
        alpha = kwargs.get('alpha', 1.0)
        fixed_vocab = bool(kwargs.get('fixed_vocab', False))

        inputs, outputs, labels, _ = generator.generate_training_set(
            self.max_input_size, batch_size, max_lookahead,
            max_edges, reserved_inputs, distance_from_start,
            max_prefix_vertices, True, alpha, fixed_vocab
        )

        parsed_examples = self._parse_symbolic_output(inputs, outputs, labels)

        nl_examples = []
        for ex in parsed_examples:
            nl_ex = self._generate_search_nl(ex, kwargs.get('use_different_names', False))
            nl_examples.append(nl_ex)

        return nl_examples

    def generate_batch_with_symbolic(self, batch_size: int, **kwargs) -> Tuple[List[NLExample], np.ndarray]:
        max_lookahead = kwargs.get('max_lookahead', 5)
        max_edges = kwargs.get('max_edges', (self.max_input_size - 5) // 3)
        reserved_inputs = kwargs.get('reserved_inputs', set())
        distance_from_start = kwargs.get('distance_from_start', -1)
        max_prefix_vertices = kwargs.get('max_prefix_vertices', self.max_input_size)
        alpha = kwargs.get('alpha', 1.0)
        fixed_vocab = bool(kwargs.get('fixed_vocab', False))

        inputs, outputs, labels, _ = generator.generate_training_set(
            self.max_input_size, batch_size, max_lookahead,
            max_edges, reserved_inputs, distance_from_start,
            max_prefix_vertices, True, alpha, fixed_vocab
        )

        parsed_examples = self._parse_symbolic_output(inputs, outputs, labels)
        nl_examples = []

        for ex in parsed_examples:
            nl_ex = self._generate_search_nl(ex, kwargs.get('use_different_names', False))
            nl_examples.append(nl_ex)

        return nl_examples, inputs
