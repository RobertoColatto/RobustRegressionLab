# exibe funções polinomiais em formato latex
def polynomial_to_latex(beta, hat=False):
    terms = []

    for i, coeff in enumerate(beta):
        if coeff == 0:
            continue

        sign = "+" if coeff >= 0 else "-"
        coeff_abs = abs(coeff)

        if i == 0:
            term = f"{coeff:.4f}"
        elif i == 1:
            term = f"{sign} {coeff_abs:.4f}x"
        else:
            term = f"{sign} {coeff_abs:.4f}x^{i}"

        terms.append(term)

    lhs = r"\hat{y}" if hat else "y"

    # remove o '+' inicial, se houver
    equation = " ".join(terms)
    if equation.startswith("+ "):
        equation = equation[2:]

    return rf"${lhs} = {equation}$"