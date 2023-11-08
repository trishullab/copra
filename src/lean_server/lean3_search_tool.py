#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
import typing
import re

class Lean3Lemma(object):
    def __init__(self, namespace: str, name: str, dfn: str) -> None:
        self.namespace = namespace
        self.name = name
        self.dfn = dfn
    
    def __str__(self) -> str:
        return f"[{self.namespace}] {self.name}: {self.dfn}"

class Lean3SearchTool(object):
    theorem_lemma_search_regex = r"(theorem|lemma) ([\w+|\d+]*) ([\S|\s]*?):="
    def __init__(self) -> None:
        self._namespaces_to_theorems : typing.Dict[str, typing.List[Lean3Lemma]] = {}
        self._theorem_lemma_search_regex = re.compile(Lean3SearchTool.theorem_lemma_search_regex)
    
    @property
    def lemmas(self) -> typing.List[Lean3Lemma]:
        return [lemma for lemmas in self._namespaces_to_theorems.values() for lemma in lemmas]
    
    @property
    def namespaces(self) -> typing.List[str]:
        return list(self._namespaces_to_theorems.keys())
    
    def add(self, lean_file_path: str, namespace: str) -> None:
        assert os.path.isfile(lean_file_path)
        assert lean_file_path.endswith(".lean")
        with open(lean_file_path, "r") as f:
            text = f.read()
            # Remove comments
            text = self._remove_comments(text)
            matches = self._theorem_lemma_search_regex.findall(text)
            for match in matches:
                name = str(match[1]).strip()
                dfn = str(match[2]).strip()
                lemma = Lean3Lemma(namespace, name, dfn)
                if namespace not in self._namespaces_to_theorems:
                    self._namespaces_to_theorems[namespace] = []
                self._namespaces_to_theorems[namespace].append(lemma)
    
    def _remove_comments(self, text: str) -> str:
        # Remove comments
        #1. First remove all nested comments
        #2. Then remove all single line comments
        # Comments are of the form:
        # 1. /- ... -/
        # 2. -- ...
        # Let's do 1
        # First, let's find all the comments
        new_text = []
        idx = 0
        while idx < len(text):
            if text[idx] == '/' and text[idx+1] == '-':
                # We found a comment
                # Find the end of the comment
                end_of_comment_idx = idx + 2
                while end_of_comment_idx < len(text) and \
                    not (text[end_of_comment_idx] == '-' and \
                    end_of_comment_idx + 1 < len(text) and \
                    text[end_of_comment_idx + 1] == '/'):
                    end_of_comment_idx += 1
                # Remove the comment
                idx = end_of_comment_idx + 2
            if idx < len(text):
                new_text.append(text[idx])
                idx += 1
        text = "".join(new_text)
        new_text = []
        # Now let's do 2
        idx = 0
        while idx < len(text):
            if text[idx] == '-' and text[idx+1] == '-':
                # We found a comment
                # Find the end of the comment
                end_of_comment_idx = idx + 2
                while end_of_comment_idx < len(text) and text[end_of_comment_idx] != '\n':
                    end_of_comment_idx += 1
                # Remove the comment
                idx = end_of_comment_idx
            if idx < len(text):
                new_text.append(text[idx])
                idx += 1
        text = "".join(new_text)
        return text
    
    def keep_only_namespaces(self, namespaces: typing.List[str]) -> None:
        namespaces_to_match = [ns.strip('.') for ns in namespaces if len(ns.strip('.')) > 0]
        namespaces_to_match = [ns + "." for ns in namespaces_to_match]
        namespaces_to_match = list(set(namespaces_to_match))
        namespaces = [ns for ns in self._namespaces_to_theorems.keys() if any([ns.startswith(n) or ns.endswith(n[:-1]) for n in namespaces_to_match])]
        self._namespaces_to_theorems = {ns: self._namespaces_to_theorems[ns] for ns in namespaces}
    
    # def transform_namespace(self, namespace: str) -> None:
    #     match_end = namespace.strip('.') + "."
    #     matching_namespaces = [ns for ns in self.namespaces if ns.startswith(match_end) or ns.endswith(match_end)]
    #     if len(matching_namespaces) == 0:
    #         print(f"WARNING: No matching namespaces found for {namespace}")
    #         return
    #     assert len(matching_namespaces) == 1, f"Multiple matching namespaces found: {matching_namespaces}"
    #     for old_namespace in matching_namespaces:
    #         # Change the matching namespace to the new namespace
    #         lemmas_in_namespace = self._namespaces_to_theorems[old_namespace]
    #         del self._namespaces_to_theorems[old_namespace]
    #         namespace = old_namespace[len(old_namespace):]
    #         namespace = namespace.strip('.')
    #         # Change the namespace of each theorem
    #         for theorem in lemmas_in_namespace:
    #             theorem.namespace = namespace
    #         if namespace not in self._namespaces_to_theorems:
    #             self._namespaces_to_theorems[namespace] = lemmas_in_namespace
    #         else:
    #             self._namespaces_to_theorems[namespace] += lemmas_in_namespace

if __name__ == "__main__":
    lean3_search_tool = Lean3SearchTool()
    mathlib_src_path = "data/test/lean_proj/_target/deps/mathlib/src"
    for root, dirs, files in os.walk(mathlib_src_path):
        for file in files:
            if file.endswith(".lean"):
                namespace = root.split(mathlib_src_path)[1].strip("/")
                namespace = namespace.replace("/", ".")
                namespace = namespace + "." + file.replace(".lean", "")
                lean3_search_tool.add(os.path.join(root, file), namespace)
    namespaces = [
        "algebra.algebra.basic",
        "algebra.order.floor",
        "algebra.associated",
        "algebra.big_operators.basic",
        "algebra.big_operators.enat",
        "algebra.big_operators.order",
        "algebra.big_operators.pi",
        "algebra.geom_sum",
        "algebra.group.pi",
        "algebra.group.commute",
        "algebra.group_power.basic",
        "algebra.group_power.identities",
        "algebra.order.floor",
        "algebra.quadratic_discriminant",
        "algebra.ring.basic",
        "analysis.asymptotics.asymptotic_equivalent",
        "analysis.mean_inequalities",
        "analysis.normed_space.basic",
        "analysis.inner_product_space.basic",
        "analysis.inner_product_space.euclidean_dist",
        "analysis.normed_space.pi_Lp",
        "analysis.special_functions.exp",
        "analysis.special_functions.exp_deriv",
        "analysis.special_functions.log",
        "analysis.special_functions.logb",
        "analysis.special_functions.log_deriv",
        "analysis.special_functions.pow",
        "analysis.special_functions.sqrt",
        "analysis.special_functions.trigonometric.basic",
        "analysis.special_functions.trigonometric.complex",
        "combinatorics.simple_graph.basic",
        "data.complex.basic",
        "data.complex.exponential",
        "data.finset.basic",
        "data.fintype.card",
        "data.int.basic",
        "data.int.gcd",
        "data.int.modeq",
        "data.int.parity",
        "data.list.intervals",
        "data.list.palindrome",
        "data.multiset.basic",
        "data.nat.basic",
        "data.nat.choose.basic",
        "data.nat.digits",
        "data.nat.factorial.basic",
        "data.nat.fib",
        "data.nat.modeq",
        "data.nat.multiplicity",
        "data.nat.parity",
        "data.nat.prime",
        "data.pnat.basic",
        "data.pnat.prime",
        "data.polynomial",
        "data.polynomial.basic",
        "data.polynomial.eval",
        "data.rat.basic",
        "data.real.basic",
        "data.real.ennreal",
        "data.real.irrational",
        "data.real.nnreal",
        "data.real.sqrt",
        "data.real.golden_ratio",
        "data.set.finite",
        "data.sym.sym2",
        "data.zmod.basic",
        "dynamics.fixed_points.basic",
        "field_theory.finite.basic",
        "geometry.euclidean.basic",
        "geometry.euclidean.circumcenter",
        "geometry.euclidean.monge_point",
        "geometry.euclidean.sphere",
        "init.data.nat.gcd",
        "linear_algebra.affine_space.affine_map",
        "linear_algebra.affine_space.independent",
        "linear_algebra.affine_space.ordered",
        "linear_algebra.finite_dimensional",
        "logic.equiv.basic",
        "measure_theory.integral.interval_integral",
        "number_theory.arithmetic_function",
        "number_theory.legendre_symbol.quadratic_reciprocity",
        "number_theory.primes_congruent_one",
        "order.bounds",
        "order.filter.basic",
        "order.well_founded",
        "topology.basic",
        "topology.instances.nnreal"
    ]
    print("Lemmas:", len(lean3_search_tool.lemmas))
    print("Namespaces:", len(lean3_search_tool.namespaces))
    print("Actual Namespaces:", len(namespaces))
    lean3_search_tool.keep_only_namespaces(namespaces)
    print("Lemmas:", len(lean3_search_tool.lemmas))
    print("Namespaces:", len(lean3_search_tool.namespaces))
    # print all theorems
    cnt_nat = 0
    cnt_real = 0
    for theorem in lean3_search_tool.lemmas:
        if "data.nat" in theorem.namespace:
            print(theorem)
            cnt_nat += 1
        if "data.real" in theorem.namespace:
            print(theorem)
            cnt_real += 1
        if cnt_nat > 10 and cnt_real > 10:
            break