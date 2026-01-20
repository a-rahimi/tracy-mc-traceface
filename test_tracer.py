# %%
from typing import NamedTuple
from numba import njit
from importlib import reload
import tracer

reload(tracer)


class User(NamedTuple):
    metric1: tracer.Float
    metric2: tracer.Float
    metric3: tracer.Int


class TestMakeSimplestDecision:
    @staticmethod
    def make_simplest_decision(user: User) -> bool:
        return user.metric3 > 5

    traced_make_simplest_decision = tracer.trace(make_simplest_decision.__func__)

    def test_make_simplest_decision_case_1(self):
        u = User(metric1=0.6, metric2=0.6, metric3=6)
        result = TestMakeSimplestDecision.traced_make_simplest_decision(u)
        assert result is self.make_simplest_decision(u)
        print("Input:", u)
        print("Result:", result)

        ir = TestMakeSimplestDecision.traced_make_simplest_decision.trace.to_ir()
        assert isinstance(ir, tracer.Return)
        assert ir.expression.text == "(metric3 > 5)"
        assert ir.expression.value is True
        assert str(ir) == "return (metric3 > 5)"

    def test_make_simplest_decision_case_2(self):
        u = User(metric1=0.6, metric2=0.6, metric3=1)
        result = TestMakeSimplestDecision.traced_make_simplest_decision(u)
        assert result is self.make_simplest_decision(u)
        print("Input:", u)
        print("Result:", result)

        ir = TestMakeSimplestDecision.traced_make_simplest_decision.trace.to_ir()
        assert isinstance(ir, tracer.Return)
        assert ir.expression.text == "(metric3 > 5)"
        assert ir.expression.value is False
        assert str(ir) == "return (metric3 > 5)"


class TestMakeDecision:
    @staticmethod
    def make_decision(user: User) -> bool:
        return user.metric1 > 0.3 and user.metric2 > 0.4 and user.metric3 > 5

    traced_make_decision = tracer.trace(make_decision.__func__)

    def test_make_decision_case_1(self):
        u = User(metric1=0.6, metric2=0.6, metric3=1)
        result = TestMakeDecision.traced_make_decision(u)
        assert result is self.make_decision(u)

        ir = TestMakeDecision.traced_make_decision.trace.to_ir()
        assert isinstance(ir, tracer.Conditional)
        assert ir.condition.text == "(metric1 > 0.3000)"
        assert ir.condition.value is True

        expected_str = """if (metric1 > 0.3000) (=True):
  if (metric2 > 0.4000) (=True):
    return (metric3 > 5)"""
        assert str(ir) == expected_str

        ir = ir.value
        assert isinstance(ir, tracer.Conditional)
        assert ir.condition.text == "(metric2 > 0.4000)"
        assert ir.condition.value is True

        ir = ir.value
        assert isinstance(ir, tracer.Return)
        assert ir.expression.text == "(metric3 > 5)"
        assert ir.expression.value is False

    def test_make_decision_case_2(self):
        u = User(metric1=0.4, metric2=0.6, metric3=1)
        result = TestMakeDecision.traced_make_decision(u)
        assert result is self.make_decision(u)

        ir = TestMakeDecision.traced_make_decision.trace.to_ir()
        assert isinstance(ir, tracer.Conditional)
        assert ir.condition.text == "(metric1 > 0.3000)"
        assert ir.condition.value is True

        expected_str = """if (metric1 > 0.3000) (=True):
  if (metric2 > 0.4000) (=True):
    return (metric3 > 5)"""
        assert str(ir) == expected_str

        ir = ir.value
        assert isinstance(ir, tracer.Conditional)
        assert ir.condition.text == "(metric2 > 0.4000)"
        assert ir.condition.value is True

        ir = ir.value
        assert isinstance(ir, tracer.Return)
        assert ir.expression.text == "(metric3 > 5)"
        assert ir.expression.value is False


class TestMakeDecisionNestedIfs:
    @staticmethod
    def make_decision_nested_ifs(user: User) -> bool:
        if user.metric1 > 0.3:
            if user.metric2 > 0.4:
                if user.metric3 > 5:
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    traced_make_decision_nested_ifs = tracer.trace(make_decision_nested_ifs.__func__)

    def test_nested_ifs_case_1(self):
        u = User(metric1=0.6, metric2=0.6, metric3=6)
        result = TestMakeDecisionNestedIfs.traced_make_decision_nested_ifs(u)
        assert result is self.make_decision_nested_ifs(u)

        ir = TestMakeDecisionNestedIfs.traced_make_decision_nested_ifs.trace.to_ir()

        expected_str = """if (metric1 > 0.3000) (=True):
  if (metric2 > 0.4000) (=True):
    if (metric3 > 5) (=True):
      return True"""
        assert str(ir) == expected_str

        assert isinstance(ir, tracer.Conditional)
        assert ir.condition.text == "(metric1 > 0.3000)"

        ir = ir.value
        assert isinstance(ir, tracer.Conditional)
        assert ir.condition.text == "(metric2 > 0.4000)"

        ir = ir.value
        assert isinstance(ir, tracer.Conditional)
        assert ir.condition.text == "(metric3 > 5)"

        ir = ir.value
        assert isinstance(ir, tracer.Return)
        assert ir.expression.text == "True"
        assert ir.expression.value is True

    def test_nested_ifs_case_2(self):
        u = User(metric1=0.4, metric2=0.6, metric3=1)
        result = TestMakeDecisionNestedIfs.traced_make_decision_nested_ifs(u)
        assert result is self.make_decision_nested_ifs(u)

        ir = TestMakeDecisionNestedIfs.traced_make_decision_nested_ifs.trace.to_ir()

        expected_str = """if (metric1 > 0.3000) (=True):
  if (metric2 > 0.4000) (=True):
    if (metric3 > 5) (=False):
      return False"""
        assert str(ir) == expected_str

        assert isinstance(ir, tracer.Conditional)
        assert ir.condition.text == "(metric1 > 0.3000)"

        ir = ir.value
        assert isinstance(ir, tracer.Conditional)
        assert ir.condition.text == "(metric2 > 0.4000)"

        ir = ir.value
        assert isinstance(ir, tracer.Conditional)
        assert ir.condition.text == "(metric3 > 5)"

        ir = ir.value
        assert isinstance(ir, tracer.Return)
        assert ir.expression.text == "False"
        assert ir.expression.value is False


class TestMakeDecisionHierarchically:
    @staticmethod
    @njit
    def _compute_weighted_score(user: User) -> float:
        return user.metric1 * 0.7 + user.metric2 * 0.3

    @staticmethod
    @njit
    def _is_safe_to_proceed(score: float) -> bool:
        if score < 0.2:
            return False
        return True

    @staticmethod
    @njit
    def _check_system_status() -> bool:
        # Condition that does not depend on User
        maintenance_mode = False
        if maintenance_mode:
            return False
        return True

    @staticmethod
    def make_decision_hierarchically(user: User) -> bool:
        if not TestMakeDecisionHierarchically._check_system_status():
            return False

        weighted_score = TestMakeDecisionHierarchically._compute_weighted_score(user)

        if not TestMakeDecisionHierarchically._is_safe_to_proceed(weighted_score):
            return False

        if weighted_score > 0.8:
            return True

        # Cascade another check dependent on original values
        if user.metric3 > 5:
            return True

        return False

    traced_make_decision_hierarchically = tracer.trace(
        make_decision_hierarchically.__func__
    )

    def test_high_weighted_score(self):
        u = User(metric1=0.9, metric2=0.9, metric3=1)
        result = TestMakeDecisionHierarchically.traced_make_decision_hierarchically(u)
        assert result is self.make_decision_hierarchically(u)

        ir = TestMakeDecisionHierarchically.traced_make_decision_hierarchically.trace.to_ir()
        print(ir)

        # Concrete boolean conditionals (False/True) should be removed from IR
        # So we skip past them and go directly to the first non-concrete conditional
        # if score < 0.2 (=False)
        assert isinstance(ir, tracer.Conditional)
        assert "< 0.2000" in ir.condition.text
        assert ir.condition.value is False

        expected_str = """if (((metric1 * 0.7000) + (metric2 * 0.3000)) < 0.2000) (=False):
  if (((metric1 * 0.7000) + (metric2 * 0.3000)) > 0.8000) (=True):
    return True"""
        assert str(ir) == expected_str

        ir = ir.value
        # if score > 0.8 (=True)
        assert isinstance(ir, tracer.Conditional)
        assert "> 0.8000" in ir.condition.text
        assert ir.condition.value is True

        ir = ir.value
        # return True
        assert isinstance(ir, tracer.Return)
        assert ir.expression.text == "True"
        assert ir.expression.value is True

    def test_medium_score_metric3_high(self):
        # Case: Medium score, safe, metric3 high
        u = User(metric1=0.5, metric2=0.5, metric3=6)
        result = TestMakeDecisionHierarchically.traced_make_decision_hierarchically(u)
        assert result is self.make_decision_hierarchically(u)

        ir = TestMakeDecisionHierarchically.traced_make_decision_hierarchically.trace.to_ir()
        # Concrete boolean conditionals are removed, so we start with score < 0.2
        # if score < 0.2 (=False)
        assert isinstance(ir, tracer.Conditional)
        assert "< 0.2000" in ir.condition.text
        assert ir.condition.value is False

        expected_str = """if (((metric1 * 0.7000) + (metric2 * 0.3000)) < 0.2000) (=False):
  if (((metric1 * 0.7000) + (metric2 * 0.3000)) > 0.8000) (=False):
    if (metric3 > 5) (=True):
      return True"""
        assert str(ir) == expected_str

        ir = ir.value
        # if score > 0.8 (=False)
        assert isinstance(ir, tracer.Conditional)
        assert "> 0.8000" in ir.condition.text
        assert ir.condition.value is False

        ir = ir.value
        # if metric3 > 5 (=True)
        assert isinstance(ir, tracer.Conditional)
        assert "metric3 > 5" in ir.condition.text
        assert ir.condition.value is True

        ir = ir.value
        # return True
        assert isinstance(ir, tracer.Return)
        assert ir.expression.text == "True"
        assert ir.expression.value is True


class TestConcreteBooleanConditions:
    def test_concrete_false_condition_removed(self):
        """Test that conditionals with concrete False values are removed from IR"""

        @tracer.trace
        def func_with_false_condition(user: User) -> bool:
            if False:
                return True
            return False

        u = User(metric1=0.5, metric2=0.5, metric3=6)
        result = func_with_false_condition(u)
        assert result is False

        ir = func_with_false_condition.trace.to_ir()
        # The conditional with False should be removed, so we should just have a Return
        assert isinstance(ir, tracer.Return)
        assert ir.expression.text == "False"
        assert ir.expression.value is False

    def test_concrete_true_condition_removed(self):
        """Test that conditionals with concrete True values are removed from IR"""

        @tracer.trace
        def func_with_true_condition(user: User) -> bool:
            if True:
                return user.metric3 > 5
            return False

        u = User(metric1=0.5, metric2=0.5, metric3=6)
        result = func_with_true_condition(u)
        assert result is True

        ir = func_with_true_condition.trace.to_ir()
        # The conditional with True should be removed, so we should just have a Return
        assert isinstance(ir, tracer.Return)
        assert "metric3 > 5" in ir.expression.text
        assert ir.expression.value is True

    def test_expression_condition_preserved(self):
        """Test that conditionals with expression conditions (not concrete booleans) are preserved"""

        @tracer.trace
        def func_with_expression_condition(user: User) -> bool:
            if user.metric1 > 0.3:
                return True
            return False

        u = User(metric1=0.5, metric2=0.5, metric3=6)
        result = func_with_expression_condition(u)
        assert result is True

        ir = func_with_expression_condition.trace.to_ir()
        # The conditional with an expression should be preserved
        assert isinstance(ir, tracer.Conditional)
        assert "(metric1 > 0.3000)" in ir.condition.text
        assert ir.condition.value is True
        assert isinstance(ir.value, tracer.Return)


class TestAssessPatient:
    class Patient(NamedTuple):
        blood_pressure: tracer.Float
        heart_rate: tracer.Int
        temperature: tracer.Float

    @staticmethod
    @njit
    def calculate_risk_score(
        blood_pressure: tracer.Float, heart_rate: tracer.Int
    ) -> tracer.Float:
        """Calculate a risk score based on blood pressure and heart rate."""
        pressure_factor = blood_pressure / 100.0
        rate_factor = float(heart_rate) / 80.0
        return pressure_factor * rate_factor

    @staticmethod
    @njit
    def is_critical_temperature(temperature: tracer.Float) -> bool:
        """Check if temperature indicates a critical condition."""
        return temperature > 37.5

    @staticmethod
    @tracer.trace
    def assess_patient(patient) -> bool:
        bp, hr, temp = patient.blood_pressure, patient.heart_rate, patient.temperature
        risk_score = TestAssessPatient.calculate_risk_score(bp, hr)

        if risk_score > 1.5:
            if hr > 100:
                return TestAssessPatient.is_critical_temperature(temp)
        return False

    def test_assess_patient_case_1(self):
        """Test the example from README with critical patient"""
        patient = self.Patient(blood_pressure=130.0, heart_rate=110, temperature=38.0)
        result = TestAssessPatient.assess_patient(patient)

        # Verify correctness
        # risk_score = (130.0 / 100.0) * (110 / 80.0) = 1.3 * 1.375 = 1.7875 > 1.5
        # hr = 110 > 100
        # temperature = 38.0 > 37.5
        # So result should be True
        assert result is True

        # Verify trace structure
        ir = TestAssessPatient.assess_patient.trace.to_ir()
        print("Trace IR:", ir)

        # The trace should show the conditional logic
        assert isinstance(ir, tracer.Conditional)
