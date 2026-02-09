## Music Off-Policy Evaluation Benchmark

Music Off-Policy Evaluation Benchmark is a dataset designed for Off-Policy Evaluation (OPE) research. It contains logged interactions from the home page of Amazon Music.

###### Use cases:

- Benchmarking OPE estimators
- Evaluating counterfactual ranking policies offline

## Security
See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License
Music Off-Policy Evaluation Benchmark © 2026 by Amazon is licensed under Creative Commons Attribution-NonCommercial 4.0 International.

## Download

Folder structure:

* `s3://music-off-policy-evaluation-benchmark/`
  * README.md
  * LICENSE.md
  * data/
    * D1/
    * D2/

You can either download files using their http link, e.g., https://s3.amazonaws.com/music-off-policy-evaluation-benchmark/LICENSE or interact with the S3 bucket using the [AWS CLI](https://aws.amazon.com/cli/). For example, to download the training dataset you can run:

```
aws s3 cp --no-sign-request --recursive s3://music-off-policy-evaluation-benchmark/ .
```

## Schema

The following table provides an overview of the dataset schema. Let it be $k$ the number of available actions for a given observation.

| Column                   | Type              | Dimension                  | Description                                                  |
| ------------------------ | ----------------- | -------------------------- | ------------------------------------------------------------ |
| actions                  | List[List[float]] | $L \times 129$ | Context vectors of size $129$ for each action in the observation. |
| rewards                  | List[float]       | $min(L, 50)$ | Observed binary rewards per action. |
| logging_selected_actions | List[int]         | $min(L, 50)$ | Selected actions of the logging policy $\pi_0$.              |
| target_selected_actions  | List[int]         | $min(L, 50)$ | Selected actions of the target policy $\pi$.                 |
| propensities             | List[List[float]] | $min(L, 50) \times min(L, 50)$ | Squared matrix of propensities. |

#### Actions

Vectors of size 129 describing a context vector. The number of available actions $L$ differs across observations.

**Example:**

Assuming for a given observation there are $L = 3$ actions, then `actions` contains $3$ vectors each of size $129$ as following:

```python
actions = [
  [1.         0.         0.28867126 ... 0.09611709 0.32168165 0.        ],
  [1.         0.         0.32252225 ... 0.19847895 0.24163522 0.        ],
  [1.         0.         0.57331926 ... 0.15339291 0.57508302 0.        ]
]
```

#### Logging selected actions

Vector of action indices selected by $\pi_0$ in the order they are displayed.

**Example:**

Let's assume in a given observation there are $L = 3$ actions such that:

```python
logging_selected_actions = [2, 1, 0]
````
It indicates that the third action (index 2) is ranked by $\pi_0$ in the first position, the second action (index 1) is ranked in second position and first action (index 0) ranked in third position.

#### Target selected actions

Vector of action indices selected by $\pi$.

**Example:**

Let's assume in a given observation there are $L = 3$ actions such that:

```python
logging_selected_actions = [2, 1, 0]
target_selected_actions = [1, 2, 0]
```

It indicates that $\pi$ ranks in first position the action with index 1 (while ranked in second position under $\pi_0$); action with index 2 is ranked by $\pi$ in second position (first position under $\pi_0$). Action index 0 is being ranked in third position under both $\pi_0$ and $\pi$.

#### Rewards

Vector of binary rewards ordered by `logging_selected_actions`.

**Example:**

Assuming we have $3$ actions in the actions set:

```python
logging_selected_actions = [2, 1, 0]
rewards = [1.0, 0.0, 0.0]
```
It indicates that a positive reward of $1$ has been observed for the action at index $2$ in `actions` vector which is ranked by $\pi_0$ in first position.

#### Propensities
Propensity matrix $P \in R^{d \times d}$ where $d = min(L, 50)$ that describes the probabilities with which actions are ranked in different positions under the logging policy $\pi_0$. Rows correspond to actions (in the order they were ranked) and columns correspond to positions in the ranking. Specifically, $P_{ij}$ describes:



$$
P_{ij}=
\begin{cases}
\text{likelihood of ranking action } i, & \text{for } i = j.\\
\text{likelihood of ranking the action in position } j \text{ instead of } i,  & \text{for } i \ne j.\\
\end{cases}
$$

**Example:**

Let it be the action space $\mathcal{A} = \{A, B, C\}$ with $L=3$ and assume:

1. The stochastic logging policy $\pi_0$ produces the ranking  $[B, A, C]$, and
2. The probability (as determined by the policy's selection mechanism) for
   
   - action $B$ to end up in the first position was $0.5$ (it could have ended up in the second position with probability $0.4$ or in the third position with probability $0.1$),
   
   - action $A$ to end up in the second position was $0.3$ (it could have ended up in the first position with probability $0.3$ or in the third position with probability $0.4$),
   
   - action $C$ to end up in the third position was $0.5$ (it could have ended up in the first position with probability $0.2$ or in the second position with probability $0.3$).
   

Then, the propensities matrix is:
```math
P = \begin{bmatrix}
       0.5 & 0.4 & 0.1 \\[0.3em]
       0.3 & 0.3 & 0.4 \\[0.3em]
       0.2 & 0.3 & 0.5
     \end{bmatrix}
```
where the rows correspond to the actions as ranked (i.e., $[B, A, C]$), and columns correspond to positions the actions could have been ranked (first, second and third).


## Citation
Paper under review.
