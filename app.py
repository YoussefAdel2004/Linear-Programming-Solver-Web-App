import streamlit as st
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from itertools import combinations

st.set_page_config(page_title="Optimization Solver", layout="wide")
st.title(" Linear Programming Solver")


def clear_all():

    st.session_state.num_vars = 2
    st.session_state.num_constraints = 2
    st.session_state.opt_type = "Maximize"
    st.session_state.allow_negative = False  

    # reset all dynamic inputs 
    for key in list(st.session_state.keys()):
        if key.startswith("Coefficient x") or key.startswith("a") or key.startswith("b"):
            st.session_state[key] = 0.0

#  sidebar 
st.sidebar.header("Settings")
st.sidebar.button(" Clear All", on_click=clear_all)

# no. of variables
if "num_vars" not in st.session_state:
    num_vars = st.sidebar.number_input("Number of Variables", 1, 10, value=2, key="num_vars")
else:
    num_vars = st.sidebar.number_input("Number of Variables", 1, 10, key="num_vars")

# no. of constraints
if "num_constraints" not in st.session_state:
    num_constraints = st.sidebar.number_input("Number of Constraints", 1, 10, value=2, key="num_constraints")
else:
    num_constraints = st.sidebar.number_input("Number of Constraints", 1, 10, key="num_constraints")

# Optimization type
if "opt_type" not in st.session_state:
    opt_type = st.sidebar.radio("Optimization Type", ["Maximize", "Minimize"], index=0, key="opt_type")
else:
    opt_type = st.sidebar.radio("Optimization Type", ["Maximize", "Minimize"], key="opt_type")

# Allow Negative 
if "allow_negative" not in st.session_state or not isinstance(st.session_state.allow_negative, bool):
    st.session_state.allow_negative = False
allow_negative = st.sidebar.checkbox(
    "Allow Negative Variables",
    value=st.session_state.allow_negative,
    key="allow_negative"
)

#  Objective Function 
st.subheader("Objective Function")
cols = st.columns(num_vars)
c = []
for i in range(num_vars):
    c.append(cols[i].number_input(
        f"Coefficient x{i+1}",
        value=st.session_state.get(f"Coefficient x{i+1}", 0.0),
        key=f"Coefficient x{i+1}"
    ))
    
c = np.array(c)
if opt_type == "Maximize":
    c = -c

#  Constraints 
st.subheader("Constraints")
A_ub, b_ub = [], []
A_eq, b_eq = [], []

for i in range(num_constraints):
    st.write(f"Constraint {i+1}")
    cols = st.columns(num_vars + 2)
    row = []
    for j in range(num_vars):
        row.append(cols[j].number_input(
            f"a{i+1}{j+1}",
            value=st.session_state.get(f"a{i}{j}", 0.0),
            key=f"a{i}{j}"
        ))
    sign = cols[num_vars].selectbox(
        "", ["<=", ">=", "="], index=0, key=f"s{i}"
    )
    rhs = cols[num_vars + 1].number_input(
        "b", value=st.session_state.get(f"b{i}", 0.0), key=f"b{i}"
    )

    if sign == "<=":
        A_ub.append(row)
        b_ub.append(rhs)
    elif sign == ">=":
        A_ub.append([-x for x in row])
        b_ub.append(-rhs)
    else:
        A_eq.append(row)
        b_eq.append(rhs)

#  Helper: Feasible Vertices 
def feasible_vertices(A_ub, b_ub, A_eq, b_eq):
    vertices = []
    constraints = []
    for r, b in zip(A_ub, b_ub):
        constraints.append((r, b))
    for r, b in zip(A_eq, b_eq):
        constraints.append((r, b))

    for (a1, b1), (a2, b2) in combinations(constraints, 2):
        A = np.array([a1, a2])
        B = np.array([b1, b2])
        if abs(np.linalg.det(A)) < 1e-9:  # Skip parallel or identical lines
            continue
        try:
            x, y = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            continue

        ok = True
        for r, b in zip(A_ub, b_ub):
            if r[0]*x + r[1]*y > b + 1e-6:
                ok = False
        for r, b in zip(A_eq, b_eq):
            if abs(r[0]*x + r[1]*y - b) > 1e-6:
                ok = False
        if ok:
            vertices.append([x, y])
    return np.array(vertices)

# ----------------- Solve -----------------
if st.button(" Solve"):
    bounds = [(None, None)] * num_vars if allow_negative else [(0, None)] * num_vars

    res = linprog(
        c,
        A_ub=A_ub if A_ub else None,
        b_ub=b_ub if b_ub else None,
        A_eq=A_eq if A_eq else None,
        b_eq=b_eq if b_eq else None,
        bounds=bounds,
        method="highs"
    )

    if res.success:
        z = -res.fun if opt_type == "Maximize" else res.fun
        st.success("✅ Optimal Solution Found")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Optimal Variables")
            for i, v in enumerate(res.x):
                st.write(f"x{i+1} = {v:.4f}")
        with col2:
            st.subheader("Objective Value")
            st.write(f"Z = {z:.4f}")

        # ----------------- Plot (only for 2 variables) -----------------
        if num_vars == 2:
            st.subheader(" Feasible Region")
            fig, ax = plt.subplots(figsize=(7, 7))
            x = np.linspace(-50, 50, 400)

            # Plot inequality constraints safely
            for r, b in zip(A_ub, b_ub):
                if abs(r[1]) > 1e-9:
                    ax.plot(x, (b - r[0]*x)/r[1])
                elif abs(r[0]) > 1e-9:
                    ax.axvline(b / r[0])

            # Plot equality constraints safely
            for r, b in zip(A_eq, b_eq):
                if abs(r[1]) > 1e-9:
                    ax.plot(x, (b - r[0]*x)/r[1], "--")
                elif abs(r[0]) > 1e-9:
                    ax.axvline(b / r[0], linestyle="--")

            # Feasible region
            verts = feasible_vertices(A_ub, b_ub, A_eq, b_eq)
            if len(verts) >= 3:
                center = verts.mean(axis=0)
                angles = np.arctan2(verts[:,1]-center[1], verts[:,0]-center[0])
                verts = verts[np.argsort(angles)]
                ax.fill(verts[:,0], verts[:,1], color="lightblue", alpha=0.4, label="Feasible Region")

                # Set axis limits
                margin = 0.15
                xmin, xmax = verts[:,0].min(), verts[:,0].max()
                ymin, ymax = verts[:,1].min(), verts[:,1].max()
                ax.set_xlim(xmin - margin*(xmax-xmin), xmax + margin*(xmax-xmin))
                ax.set_ylim(ymin - margin*(ymax-ymin), ymax + margin*(ymax-ymin))

            ax.set_xlabel("x1")
            ax.set_ylabel("x2")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)

            #  Table of Vertices and Objective 
            if len(verts) > 0:
                st.subheader(" Feasible Vertices and Objective Values")

                # Calculate objective for each vertex
                table_data = []
                for v in verts:
                    obj_val = np.dot(c if opt_type == "Minimize" else -c, v)
                    table_data.append([v[0], v[1], obj_val])

                table_data = np.array(table_data)

                # Determine max and min
                max_idx = table_data[:, 2].argmax()
                min_idx = table_data[:, 2].argmin()

                # Prepare display table
                display_table = []
                for i, row in enumerate(table_data):
                    label = ""
                    if i == max_idx:
                        label = "Max"
                    elif i == min_idx:
                        label = "Min"
                    display_table.append([row[0], row[1], row[2], label])

                st.table(
                    np.array(display_table, dtype=object)
                )

    else:
        st.error(res.message)


#  py -m streamlit run app.py
