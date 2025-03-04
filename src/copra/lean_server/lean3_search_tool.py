#!/usr/bin/env python3

import os
import typing
import re
from copra.lean_server.lean_utils import Lean3Utils

class Constants(object):
    mathlib_useful_imports = [
        'algebra.algebra.basic',
        'algebra.order.floor',
        'algebra.associated',
        'algebra.big_operators.basic',
        'algebra.big_operators.enat',
        'algebra.big_operators.order',
        'algebra.big_operators.pi',
        'algebra.geom_sum',
        'algebra.group.pi',
        'algebra.group.commute',
        'algebra.group_power.basic',
        'algebra.group_power.identities',
        'algebra.order.floor',
        'algebra.quadratic_discriminant',
        'algebra.ring.basic',
        'analysis.asymptotics.asymptotic_equivalent',
        'analysis.mean_inequalities',
        'analysis.normed_space.basic',
        'analysis.inner_product_space.basic',
        'analysis.inner_product_space.euclidean_dist',
        'analysis.normed_space.pi_Lp',
        'analysis.special_functions.exp',
        'analysis.special_functions.exp_deriv',
        'analysis.special_functions.log',
        'analysis.special_functions.logb',
        'analysis.special_functions.log_deriv',
        'analysis.special_functions.pow',
        'analysis.special_functions.sqrt',
        'analysis.special_functions.trigonometric.basic',
        'analysis.special_functions.trigonometric.complex',
        'combinatorics.simple_graph.basic',
        'data.complex.basic',
        'data.complex.exponential',
        'data.finset.basic',
        'data.fintype.card',
        'data.int.basic',
        'data.int.gcd',
        'data.int.modeq',
        'data.int.parity',
        'data.list.intervals',
        'data.list.palindrome',
        'data.multiset.basic',
        'data.nat.basic',
        'data.nat.choose.basic',
        'data.nat.digits',
        'data.nat.factorial.basic',
        'data.nat.fib',
        'data.nat.modeq',
        'data.nat.multiplicity',
        'data.nat.parity',
        'data.nat.prime',
        'data.pnat.basic',
        'data.pnat.prime',
        'data.polynomial.algebra_map',
        'data.polynomial.field_division',
        'data.polynomial.derivative',
        'data.polynomial.identities',
        'data.polynomial.integral_normalization',
        'data.polynomial.basic',
        'data.polynomial.eval',
        'data.rat.basic',
        'data.real.basic',
        'data.real.ennreal',
        'data.real.irrational',
        'data.real.nnreal',
        'data.real.sqrt',
        'data.real.golden_ratio',
        'data.set.finite',
        'data.sym.sym2',
        'data.zmod.basic',
        'dynamics.fixed_points.basic',
        'field_theory.finite.basic',
        'geometry.euclidean.basic',
        'geometry.euclidean.circumcenter',
        'geometry.euclidean.monge_point',
        'geometry.euclidean.sphere',
        'linear_algebra.affine_space.affine_map',
        'linear_algebra.affine_space.independent',
        'linear_algebra.affine_space.ordered',
        'linear_algebra.finite_dimensional',
        'logic.equiv.basic',
        'measure_theory.integral.interval_integral',
        'number_theory.arithmetic_function',
        'number_theory.legendre_symbol.quadratic_reciprocity',
        'number_theory.primes_congruent_one',
        'order.bounds',
        'order.filter.basic',
        'order.well_founded',
        'topology.basic',
        'topology.instances.nnreal'
    ]

    lean_useful_imports = [
        'init.util',
        'init.function',
        'init.coe',
        'init.ite_simp',
        'init.propext',
        'init.core',
        'init.logic',
        'init.wf',
        'init.cc_lemmas',
        'init.version',
        'init.funext',
        'init.classical',
        'init.default',
        'init.meta.derive',
        'init.meta.well_founded_tactics',
        'init.meta.interactive_base',
        'init.meta.mk_has_reflect_instance',
        'init.meta.decl_cmds',
        'init.meta.hole_command',
        'init.meta.injection_tactic',
        'init.meta.exceptional',
        'init.meta.match_tactic',
        'init.meta.interactive',
        'init.meta.mk_inhabited_instance',
        'init.meta.has_reflect',
        'init.meta.format',
        'init.meta.expr_address',
        'init.meta.type_context',
        'init.meta.task',
        'init.meta.simp_tactic',
        'init.meta.congr_lemma',
        'init.meta.ref',
        'init.meta.feature_search',
        'init.meta.congr_tactic',
        'init.meta.attribute',
        'init.meta.async_tactic',
        'init.meta.rb_map',
        'init.meta.tagged_format',
        'init.meta.expr',
        'init.meta.options',
        'init.meta.instance_cache',
        'init.meta.rewrite_tactic',
        'init.meta.set_get_option_tactics',
        'init.meta.contradiction_tactic',
        'init.meta.declaration',
        'init.meta.backward',
        'init.meta.comp_value_tactics',
        'init.meta.occurrences',
        'init.meta.fun_info',
        'init.meta.case_tag',
        'init.meta.name',
        'init.meta.json',
        'init.meta.constructor_tactic',
        'init.meta.float',
        'init.meta.mk_dec_eq_instance',
        'init.meta.module_info',
        'init.meta.vm',
        'init.meta.relation_tactics',
        'init.meta.level',
        'init.meta.pexpr',
        'init.meta.local_context',
        'init.meta.rec_util',
        'init.meta.environment',
        'init.meta.default',
        'init.meta.ac_tactics',
        'init.meta.mk_has_sizeof_instance',
        'init.meta.tactic',
        'init.meta.interaction_monad',
        'init.meta.parser',
        'init.meta.smt.ematch',
        'init.meta.smt.interactive',
        'init.meta.smt.smt_tactic',
        'init.meta.smt.rsimp',
        'init.meta.smt.default',
        'init.meta.smt.congruence_closure',
        'init.meta.converter.interactive',
        'init.meta.converter.conv',
        'init.meta.converter.default',
        'init.meta.widget.basic',
        'init.meta.widget.html_cmd',
        'init.meta.widget.replace_save_info',
        'init.meta.widget.tactic_component',
        'init.meta.widget.interactive_expr',
        'init.meta.widget.default',
        'init.algebra.classes',
        'init.algebra.order',
        'init.algebra.functions',
        'init.algebra.default',
        'init.control.functor',
        'init.control.lift',
        'init.control.lawful',
        'init.control.applicative',
        'init.control.monad',
        'init.control.except',
        'init.control.state',
        'init.control.alternative',
        'init.control.option',
        'init.control.combinators',
        'init.control.reader',
        'init.control.id',
        'init.control.default',
        'init.control.monad_fail',
        'init.data.basic',
        'init.data.to_string',
        'init.data.prod',
        'init.data.repr',
        'init.data.set',
        'init.data.punit',
        'init.data.setoid',
        'init.data.default',
        'init.data.quot',
        'init.data.char.basic',
        'init.data.char.classes',
        'init.data.char.lemmas',
        'init.data.char.default',
        'init.data.nat.gcd',
        'init.data.nat.basic',
        'init.data.nat.div',
        'init.data.nat.lemmas',
        'init.data.nat.default',
        'init.data.nat.bitwise',
        'init.data.option.basic',
        'init.data.option.instances',
        'init.data.list.basic',
        'init.data.list.instances',
        'init.data.list.qsort',
        'init.data.list.lemmas',
        'init.data.list.default',
        'init.data.string.basic',
        'init.data.string.ops',
        'init.data.string.default',
        'init.data.fin.basic',
        'init.data.fin.ops',
        'init.data.fin.default',
        'init.data.ordering.basic',
        'init.data.ordering.lemmas',
        'init.data.ordering.default',
        'init.data.array.slice',
        'init.data.array.basic',
        'init.data.array.default',
        'init.data.subtype.basic',
        'init.data.subtype.instances',
        'init.data.subtype.default',
        'init.data.unsigned.basic',
        'init.data.unsigned.ops',
        'init.data.unsigned.default',
        'init.data.bool.basic',
        'init.data.bool.lemmas',
        'init.data.bool.default',
        'init.data.sum.basic',
        'init.data.sum.instances',
        'init.data.sum.default',
        'init.data.sigma.basic',
        'init.data.sigma.lex',
        'init.data.sigma.default',
        'init.data.int.comp_lemmas',
        'init.data.int.basic',
        'init.data.int.order',
        'init.data.int.default',
        'init.data.int.bitwise',
        'smt.prove',
        'smt.array',
        'smt.arith',
        'smt.default',
        'system.random',
        'system.io_interface',
        'system.io',
        'data.buffer',
        'data.dlist',
        'data.vector',
        'data.buffer.parser',
        'tools.debugger.util',
        'tools.debugger.cli',
        'tools.debugger.default'
    ]

class Lean3Lemma(object):
    def __init__(self, namespace: str, name: str, dfn: str) -> None:
        self.namespace = namespace
        self.name = name
        self.dfn = dfn
    
    def __str__(self) -> str:
        return f"{self.namespace}.{self.name}: {self.dfn}"

class Lean3SearchTool(object):
    theorem_lemma_search_regex = r"(theorem|lemma) ([\w+|\d+]*) ([\S|\s]*?):="
    def __init__(self, mathlib_path = None, imports = None):
        assert mathlib_path is None or os.path.isdir(mathlib_path)
        self.mathlib_path = mathlib_path
        self.lean_lib_path = Lean3Utils.get_lean_lib_path()
        self._mathlib_src_path = os.path.join(self.mathlib_path, "src") if self.mathlib_path is not None else None
        self._imports = imports if imports is not None else Constants.mathlib_useful_imports + Constants.lean_useful_imports
        self._import_paths = []
        # Map the imports to the files they are imported from
        for i in range(len(self._imports)):
            possible_relative_path = self._imports[i].replace(".", "/") + ".lean"
            possible_lean_lib_path = os.path.join(self.lean_lib_path, possible_relative_path) if self.lean_lib_path is not None else None
            possible_mathlib_src_path = os.path.join(self._mathlib_src_path, possible_relative_path) if self._mathlib_src_path is not None else None
            if self.lean_lib_path is not None and os.path.isfile(possible_lean_lib_path):
                self._import_paths.append(possible_lean_lib_path)
            if self._mathlib_src_path is not None and os.path.isfile(possible_mathlib_src_path):
                self._import_paths.append(possible_mathlib_src_path)
        self._namespaces_to_theorems : typing.Dict[str, typing.List[Lean3Lemma]] = {}
        self._lemmas : typing.List[Lean3Lemma] = []
    
    def initialize(self) -> None:
        for import_path in self._import_paths:
            self._add(import_path)

    @property
    def lemmas(self) -> typing.List[Lean3Lemma]:
        return self._lemmas
    
    @property
    def namespaces(self) -> typing.List[str]:
        return list(self._namespaces_to_theorems.keys())
    
    def _add(self, lean_file_path: str) -> None:
        assert os.path.isfile(lean_file_path)
        assert lean_file_path.endswith(".lean")
        with open(lean_file_path, "r") as f:
            text = f.read()
            # Remove comments
            text = Lean3Utils.remove_comments(text)
            theorems_with_namespaces = Lean3Utils.find_theorems_with_namespaces(text)
            for namespace, name, dfn in theorems_with_namespaces:
                lemma = Lean3Lemma(namespace, name, dfn)
                if namespace not in self._namespaces_to_theorems:
                    self._namespaces_to_theorems[namespace] = []
                self._namespaces_to_theorems[namespace].append(lemma)
                self._lemmas.append(lemma)

if __name__ == "__main__":
    mathlib_path = "data/benchmarks/miniF2F/_target/deps/mathlib"
    lean3_search_tool = Lean3SearchTool(mathlib_path=mathlib_path)
    lean3_search_tool.initialize()
    print("Lemmas:", len(lean3_search_tool.lemmas))
    print("Namespaces:", len(lean3_search_tool.namespaces))
    # print all theorems
    cnt_nat = 0
    cnt_real = 0
    for theorem in lean3_search_tool.lemmas:
        if theorem.namespace.startswith("nat"):
            print(theorem)
            cnt_nat += 1
        if theorem.namespace.startswith("real"):
            print(theorem)
            cnt_real += 1
        if cnt_nat > 100 and cnt_real > 100:
            break