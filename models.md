# Process will be 

environment setup → fairness rules → algorithm settings → borrower/arm arrives → algorithm chooses action → environment generates outcome → metrics evaluate fairness and utility


## 1. EnvironmentConfig
What it does

This defines the world setup for one experiment.

It controls things like:

- total budget
- maximum rounds
- whether feedback is delayed
- whether the world changes over time
- random seed

### What kind of questions it answers
- How much money does the lender have?
- How long does the simulation run?
- Are repayments immediate or delayed?
- Is the environment stationary or drifting?

Example

```python
env_config = EnvironmentConfig(
    total_budget=100000,
    max_rounds=1000,
    delayed_feedback=True,
    non_stationary=True,
    random_seed=42
)
```

## 2. FairnessConfig
What it does

This stores the fairness rules and policy constraints the system must respect.

It answers:

- which groups must not be under-served
- what minimum allocation share is required
- what demographic parity gap is acceptable
- whether equal opportunity is enforced
- whether some groups are mandatory to serve

Example

```python
FairnessConfig(
    min_allocation_share_by_group={"rural": 0.2, "marginalized": 0.15},
    target_share_by_region={"north": 0.3, "south": 0.2},
    must_serve_groups={"isolated"},
    demographic_parity_tolerance=0.1
)
```

It is read by:

- the algorithm, to modify action selection
- the metrics/evaluation, to check whether fairness was violated
- It does not describe one borrower.
- It describes the fairness obligations of the whole system.

## 3. AlgorithmConfig
What it does

This stores the behavior settings of the algorithm.

It controls things like:

- epsilon for epsilon-greedy
- epsilon decay
- exploration budget fraction
- fairness penalty strength
- cost penalty strength
- how reward and impact are weighted

Example

```python
AlgorithmConfig(
    epsilon=0.1,
    epsilon_decay=0.995,
    exploration_budget_fraction=0.2,
    penalty_lambda=2.0,
    cost_lambda=0.5,
    social_impact_weight=0.3
)
```
How it connects to others

It is used by the bandit algorithm.

Important distinction:
- FairnessConfig says what fairness must be respected
- AlgorithmConfig says how strongly the algorithm reacts to those fairness goals

So:
fairness config = policy
algorithm config = implementation knobs

## 4. GroupMetadata
What it does

This stores information about a group itself, not a single applicant.

For example:

whether this group is mandatory to serve
its policy priority
its minimum allocation share
Example

```python
GroupMetadata(
    group_name="geographically_isolated",
    mandatory_service=True,
    policy_priority_weight=1.5,
    minimum_allocation_share=0.2,
    minimum_budget_share=0.15
)
```

This is optional, but useful when you want a separate table of group-level rules.

It connects to:

- FairnessConfig
- ApplicantContext or ContextualArmSpec

You can think of it as a dictionary of facts about groups.

## 5. ApplicantContext
This is the most important model for the contextual bandit version.

It describes the borrower/application that arrives at a round.

This includes:

requested amount
region
sector
income group
protected group
social vulnerability
impact score
indebtedness risk

```pythonApplicantContext(
    applicant_id="A102",
    requested_loan_amount=250.0,
    loan_purpose="farming",
    sector="agriculture",
    region="north",
    region_type="rural",
    business_stage="new",
    repeat_borrower=False,
    protected_group="marginalized",
    income_group="very_low",
    socially_marginalized=True,
    geographically_isolated=True,
    poverty_relief_score=0.9,
    social_impact_score=0.8,
    indebtedness_risk_score=0.3,
    priority_weight=1.4
) 
```
What it means conceptually

At round t, the environment says:

Here is the current applicant.

The algorithm sees that applicant’s context and decides what to do.

How it connects to others

It connects to:

- LoanAction, because the action is chosen based on this context
- OutcomeModelParams, because those hidden parameters determine what happens after the action
- FairnessConfig, because the applicant belongs to protected groups used in fairness checks

This is the core input to a contextual bandit.


## 6. LoanAction
What it does

This defines the decision the algorithm makes for a given applicant.

For example:

reject
approve small loan
approve medium loan
approve large loan

```python
LoanAction(
    action_id="medium_loan",
    approve=True,
    approved_loan_amount=200.0,
    loan_term_steps=6
)   
```

In a simple bandit, the action is just “choose arm 2.”

In a contextual bandit, the action is more like:

what loan offer do I make to this applicant?

So LoanAction is the output of the policy.

How it connects to others

It takes:

ApplicantContext as input for decision-making

It affects:

OutcomeModelParams, because reward/cost/delay depend on both applicant and action
budget updates in the environment
fairness metrics indirectly, because approved actions count toward allocation


## 7. OutcomeModelParams

What it does

This defines the hidden outcome-generating process.

It tells the environment how to simulate:

repayment probability
repayment delay
expected repayment amount
social impact
indebtedness change

Example

```python
OutcomeModelParams(
    repayment_probability=0.72,
    repayment_delay_steps=4,
    expected_repayment_amount=230.0,
    realized_social_impact_mean=0.8,
    realized_social_impact_std=0.1,
    indebtedness_change_mean=-0.2,
    indebtedness_change_std=0.05
)
```

Because borrower context and actual outcome generation are not always the same thing.

Example:

the algorithm may observe “rural, low-income, requested 200”
but the environment privately uses hidden parameters to determine repayment and impact

This is good design because it lets you simulate uncertainty.

How it connects to others

It works with:

ApplicantContext
LoanAction

Together, those determine:

reward
cost
delay
social outcome

So if ApplicantContext is “who arrived,” then OutcomeModelParams is “what is likely to happen after lending.”


## 8. ContextualArmSpec

This is the bridge model between simple arms and full contextual bandits.

Instead of modeling each arriving borrower separately, you model a borrower segment template.

Example segments:

low-income rural agriculture borrowers
urban retail borrowers
isolated region borrowers

Each segment has:

group markers
cost profile
repayment probability
impact scores
indebtedness risk

```python
ContextualArmSpec(
    name="rural_agri_low_income",
    protected_group="marginalized",
    region="north",
    region_type="isolated",
    income_group="very_low",
    socially_marginalized=True,
    geographically_isolated=True,
    priority_weight=1.5,
    mandatory_service_group=True,
    expected_cost=180.0,
    cost_std=30.0,
    repayment_probability=0.58,
    repayment_delay_steps=5,
    expected_repayment_amount=210.0,
    poverty_relief_score=0.95,
    social_impact_score=0.88,
    indebtedness_risk_score=0.25
)
```

Why use it

This is best when:

you want more realism than plain arms
but you are not ready to fully model applicant-by-applicant contextual arrivals
How it connects to others

It can replace ApplicantContext + OutcomeModelParams in a simpler setup.

So:

simple version: choose among ContextualArmSpec
advanced version: receive ApplicantContext, choose LoanAction, simulate via OutcomeModelParams


## Architecture: full contextual bandit design

This is more realistic.

Flow
EnvironmentConfig sets the world
FairnessConfig defines fairness constraints
AlgorithmConfig defines learning behavior
At each round, the environment emits an ApplicantContext
The algorithm observes that context
The algorithm selects a LoanAction
The environment uses OutcomeModelParams to simulate reward, delay, social impact, and debt effect
Metrics compute reward, cost, and fairness over time
Mental model
ApplicantContext = who is asking now
LoanAction = what offer do we make
OutcomeModelParams = what happens after that offer
fairness config = how to judge whether allocation was fair

This is the real contextual-bandit formulation.


## The easiest way to remember them

Context says who arrived, action says what we do, outcome says what happens, fairness says what is allowed, algorithm config says how aggressively we learn, and environment config says what world we are in.

That maps directly to:

ApplicantContext → who arrived
LoanAction → what we do
OutcomeModelParams → what happens
FairnessConfig → what is allowed
AlgorithmConfig → how we learn
EnvironmentConfig → what world we simulate
ContextualArmSpec → simplified segment-based substitute for context+outcome
