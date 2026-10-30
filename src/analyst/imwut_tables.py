"""Paper tables for the analyst, written to results/analyst/imwut_tables/.

`analyst_review` is the review-tab summary: majority-vote quality of each step,
pooled and per home, with Fleiss' κ for each home's reviewer panel.
`analyst_funnel` is the question-level funnel (Milan, Aruba, Cairo, All).
`analyst_questions` is nine questions: rotated category, full wording, and a check
or cross for whether the analyst answered. `analyst_questions_all` is the same
for all 29, for the appendix.

  python src/analyst/imwut_tables.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from analyst.config import PROJECT_ROOT, RESULTS_DIR  # noqa: E402
from analyst.rater import _fleiss  # noqa: E402
from analyst.review import _funnel, consensus_votes  # noqa: E402

RUN = RESULTS_DIR / "runs" / "analyst_questions"
REVIEW = "review_v2"
ALEX = "alexkarpekov"
MAX_RATINGS = 3
OUT = RESULTS_DIR / "imwut_tables"
HOMES = ["milan", "aruba", "cairo"]
SCOPES = ["all", *HOMES]

# (kind, key, latex label). kind is "group", "rate", or "kappa".
# rate key is the step metric. kappa key is the step number (2 includes decoys).
ROWS = [
    ("group", None, r"1.\ Is the question answerable?"),
    ("rate", "step1", r"Agrees with \ToolName{}"),
    ("rate", "step1_answerable", r"\hspace{0.4em}\ToolName{} said answerable"),
    ("rate", "step1_not_answerable", r"\hspace{0.4em}\ToolName{} said not answerable"),
    ("kappa", 1, r"\hspace{0.4em}Fleiss' $\kappa$"),
    ("group", None, r"2.\ Is the re-written query relevant"),
    ("rate", "step2", r"Planner query"),
    ("rate", "step2_decoy", r"\hspace{0.4em}Negative Examples$^{\downarrow}$"),
    ("kappa", 2, r"\hspace{0.4em}Fleiss' $\kappa$"),
    ("group", None, r"3.\ Retrieved data sample contains the behavior"),
    ("rate", "step3_top", r"Top match"),
    ("rate", "step3_random", r"Random match"),
    ("rate", "step3_negative", r"Negative Examples$^{\downarrow}$"),
    ("kappa", 3, r"\hspace{0.4em}Fleiss' $\kappa$"),
    ("group", None, r"4.\ Is the finding supported?"),
    ("rate", "step4", r"Finding"),
    ("kappa", 4, r"\hspace{0.4em}Fleiss' $\kappa$"),
]

NOTE = (
    r"Each cell is the share of decided items on which the majority vote agrees with \ToolName{}, "
    r"with yeses over decided items in parentheses. "
    r"All is the pool of items, not the mean of the homes. "
    r"A yes in step~1 means the vote matches \ToolName{}'s answerable\,/\,not verdict. "
    r"A yes in steps~2--4 means the query was judged relevant, "
    r"the retrieved sample contains the behavior, "
    r"or the finding is supported. "
    r"Rows marked $^{\downarrow}$ are controls and should be low. "
    r"REVIEWERS_CLAUSE. "
    r"Fleiss' $\kappa$ is how much the raters agree with each other."
)

_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _reviewer_clause(key: pd.DataFrame, ratings: pd.DataFrame) -> str:
    """'Milan steps 1--3 have four reviewers; ...; step~4 has one in every home', from who rated what."""
    rated = (ratings.loc[ratings["rating"].isin(["yes", "no"]), ["item_id", "reviewer"]]
             .merge(key[["item_id", "home", "step"]], on="item_id"))
    n = rated.groupby(["home", "step"])["reviewer"].nunique()
    parts = []
    for home in HOMES:
        c = [int(n.get((home, s), 0)) for s in (1, 2, 3)]
        if len(set(c)) == 1:
            parts.append(f"{home.capitalize()} steps 1--3 have {_NUM_WORDS[c[0]]} "
                         f"reviewer{'s' if c[0] != 1 else ''}")
        else:
            parts.append(f"{home.capitalize()} has " + ", ".join(
                f"{_NUM_WORDS[k]} in step~{s}" for s, k in zip((1, 2, 3), c)))
    s4 = {int(n.get((h, 4), 0)) for h in HOMES}
    if len(s4) == 1:
        k = s4.pop()
        parts.append(f"step~4 has {_NUM_WORDS[k]} reviewer{'s' if k != 1 else ''} in every home")
    else:
        parts.append("step~4 has " + ", ".join(
            f"{_NUM_WORDS[int(n.get((h, 4), 0))]} in {h.capitalize()}" for h in HOMES))
    return "; ".join(parts)


def _as_bool(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "yes"}


def _load() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    key = pd.concat([pd.read_csv(p) for p in sorted((RUN / "review" / f"{REVIEW}_key").glob("*_key.csv"))],
                    ignore_index=True)
    ratings = pd.concat([pd.read_csv(p) for p in sorted((RUN / "review" / REVIEW).glob("*.csv"))],
                        ignore_index=True)
    ratings["rating"] = ratings["rating"].astype(str).str.lower().str.strip()
    all_ratings = ratings.drop_duplicates(["reviewer", "item_id"], keep="last")
    # Alex steps aside on any item with more than MAX_RATINGS ratings, so external reviewers replace him.
    # He still breaks an unsure tie among those three (consensus_votes).
    rated = all_ratings[all_ratings["rating"].isin(["yes", "no", "unsure"])]
    crowded = rated.groupby("item_id")["reviewer"].transform("size") > MAX_RATINGS
    ratings = all_ratings.drop(rated.index[crowded & rated["reviewer"].str.startswith(ALEX)])
    key["analyst_answerable"] = key["analyst_answerable"].map(_as_bool)
    key["stratum"] = key["stratum"].fillna("").astype(str)
    consensus = consensus_votes(all_ratings, counted=ratings)
    return key, consensus, ratings


def _counts(sub: pd.DataFrame, consensus: dict, *, agree: bool) -> dict:
    yes = no = unsure = 0
    for row in sub.itertuples(index=False):
        vote = consensus.get(row.item_id)
        if vote is None:
            continue
        if agree and vote in ("yes", "no"):
            vote = "yes" if (vote == "yes") == bool(row.analyst_answerable) else "no"
        if vote == "yes":
            yes += 1
        elif vote == "no":
            no += 1
        else:
            unsure += 1
    return {"yes": yes, "no": no, "unsure": unsure}


def _metrics(key: pd.DataFrame, consensus: dict) -> dict:
    s2 = key["stratum"] != "decoy"
    groups = {
        "step1": (key[key.step == 1], True),
        "step1_answerable": (key[(key.step == 1) & key.analyst_answerable], True),
        "step1_not_answerable": (key[(key.step == 1) & ~key.analyst_answerable], True),
        "step2": (key[(key.step == 2) & s2], False),
        "step2_decoy": (key[(key.step == 2) & (key.stratum == "decoy")], False),
        "step3_top": (key[(key.step == 3) & (key.stratum == "top")], False),
        "step3_random": (key[(key.step == 3) & (key.stratum == "random")], False),
        "step3_negative": (key[(key.step == 3) & (key.stratum == "negative")], False),
        "step4": (key[key.step == 4], False),
    }
    return {name: _counts(sub, consensus, agree=agree) for name, (sub, agree) in groups.items()}


def _fleiss_by_step(key: pd.DataFrame, ratings: pd.DataFrame) -> dict:
    """Fleiss' κ per step. Step 2 includes decoys. Items need a yes/no from every reviewer who rated the step."""
    reviewers = sorted(ratings["reviewer"].unique())
    wide = ratings.pivot(index="item_id", columns="reviewer", values="rating")
    step_of = key.drop_duplicates("item_id").set_index("item_id")["step"]
    out = {}
    for s in (1, 2, 3, 4):
        sub = wide.loc[wide.index.intersection(step_of[step_of.eq(s)].index)]
        active = [rv for rv in reviewers if rv in sub.columns and sub[rv].isin(["yes", "no"]).any()]
        if len(active) < 2 or sub.empty:
            out[s] = None
            continue
        ok = sub[active].apply(lambda r: r.isin(["yes", "no"]).all(), axis=1)
        rows = sub.loc[ok, active].astype(str).values.tolist()
        k = _fleiss(rows)
        out[s] = None if not rows or k is None else {"kappa": k, "n": len(rows), "raters": len(active)}
    return out


def _rate_cell(c: dict) -> str:
    n = c["yes"] + c["no"]
    if not n:
        return r"\textemdash"
    pct = 100.0 * c["yes"] / n
    # Same size as the Fleiss κ point estimates (the table's \small). The count
    # stays smaller so the widest cells still fit in a 15% column.
    return rf"{pct:.1f}\%\hspace{{0.35em}}{{\tiny({c['yes']}/{n})}}"


def _kappa_cell(k: dict | None) -> str:
    if not k or not k["n"]:
        return r"\textemdash"
    # [t] keeps the value on the row's top line. arraystretch is set before the tabular
    # so the strut is actually tight. \tabularnewline, not \\[dimen]: arydshln (loaded in
    # the paper) breaks on an optional argument to \\.
    return (
        r"{\renewcommand{\arraystretch}{0.6}"
        r"\begin{tabular}[t]{@{}c@{}}"
        rf"{k['kappa']:.2f}\tabularnewline"
        rf"{{\tiny($n{{=}}{k['n']}$)}}"
        r"\end{tabular}}"
    )


def build_tabular(by_scope: dict, kappa_by_home: dict) -> list[str]:
    headers = " & ".join([r"\textbf{Pipeline Step}"] + [rf"\textbf{{{s.capitalize()}}}" for s in SCOPES])
    # Step is 40% of the line; All, Milan, Aruba, and Cairo are 15% each.
    # All is a p column so \columncolor fills the whole column. \tabcolsep 0 keeps the five widths on \linewidth.
    lines = [
        r"\setlength{\tabcolsep}{0pt}",
        r"\begin{tabular}{@{}w{l}{0.40\linewidth}"
        r">{\columncolor{black!5}\centering\arraybackslash}p{0.15\linewidth}"
        r"*{3}{w{c}{0.15\linewidth}}@{}}",
        r"\toprule",
        headers + r" \\",
        r"\midrule",
    ]
    for kind, key, label in ROWS:
        if kind == "group":
            # Column 1 only, so the All cell keeps its color. No \vspace here:
            # inside this p column it drops the heading below the row and opens a tall gap.
            lines.append(
                r"\multicolumn{1}{@{}>{\raggedright\arraybackslash}p{0.40\linewidth}}{\textbf{"
                + label + r"}} & & & & \\"
            )
            continue
        if kind == "rate":
            cells = [_rate_cell(by_scope[s][key]) for s in SCOPES]
            end = r" \\"
        else:
            # κ is per home. All stays blank: the homes do not share one rater panel.
            # Plain \\ : arydshln breaks on \\[dimen]. The inner arraystretch keeps the second line tight.
            cells = [
                r"\textemdash" if s == "all" else _kappa_cell((kappa_by_home.get(s) or {}).get(key))
                for s in SCOPES
            ]
            end = r" \\"
        lines.append(" & ".join([label, *cells]) + end)
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def _tex_fragment(tabular: list[str], note: str, *, label: str, env: str = "table*") -> str:
    return "\n".join([
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\caption{" + note + "}",
        rf"\label{{{label}}}",
        r"\small",
        r"\renewcommand{\arraystretch}{1.05}",
        *tabular,
        rf"\end{{{env}}}",
        "",
    ])


def _standalone(tabular: list[str], note: str, *, width_cm: float = 16.2, page_width: str | None = None) -> str:
    width = page_width or f"{width_cm}cm"
    note_width = rf"\dimexpr{width}-6pt\relax" if page_width else f"{width_cm - 0.2}cm"
    return "\n".join([
        r"\documentclass[border=8pt]{standalone}",
        r"\usepackage{booktabs}",
        r"\providecommand{\ToolName}{\textbf{SensorQ}}",
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
        r"\usepackage{array}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{graphicx}",
        r"\usepackage{multirow}",
        r"\begin{document}",
        rf"\begin{{minipage}}{{{width}}}",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.05}",
        *tabular,
        r"\vspace{6pt}",
        rf"\parbox{{{note_width}}}{{\footnotesize " + note + "}",
        r"\end{minipage}",
        r"\end{document}",
        "",
    ])


def _compile(standalone: str, pdf_path: Path, png_path: Path) -> None:
    pdflatex = shutil.which("pdflatex") or next(
        (p for p in (
            "/Library/TeX/texbin/pdflatex",
            "/usr/local/texlive/2026/bin/universal-darwin/pdflatex",
            "/usr/local/texlive/2025/bin/universal-darwin/pdflatex",
            "/usr/local/texlive/2024/bin/universal-darwin/pdflatex",
        ) if Path(p).exists()),
        None,
    )
    if not pdflatex:
        print("[WARN] pdflatex not found; wrote the .tex only", file=sys.stderr)
        return
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        tex = tmp / "table.tex"
        tex.write_text(standalone, encoding="utf-8")
        result = subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", str(tex)],
            cwd=tmp, capture_output=True, text=True,
        )
        pdf = tmp / "table.pdf"
        if not pdf.exists():
            tail = (result.stdout or result.stderr or "")[-1200:]
            print(f"[WARN] pdflatex failed:\n{tail}", file=sys.stderr)
            return
        shutil.copy(pdf, pdf_path)
        print(f"PDF  {pdf_path}")
        pdftoppm = shutil.which("pdftoppm")
        if not pdftoppm:
            print("[WARN] pdftoppm not found; PDF written, PNG skipped", file=sys.stderr)
            return
        subprocess.run(
            [pdftoppm, "-r", "300", "-png", "-singlefile", str(pdf), str(tmp / "out")],
            check=True,
        )
        shutil.copy(tmp / "out.png", png_path)
        print(f"PNG  {png_path}")


def write_review_table() -> Path:
    key, consensus, ratings = _load()
    by_scope = {"all": _metrics(key, consensus)}
    for home in HOMES:
        by_scope[home] = _metrics(key[key.home == home], consensus)
    kappa_by_home = {}
    for home in HOMES:
        ids = set(key.loc[key.home == home, "item_id"])
        kappa_by_home[home] = _fleiss_by_step(
            key[key.item_id.isin(ids)], ratings[ratings.item_id.isin(ids)])
    note = NOTE.replace("REVIEWERS_CLAUSE", _reviewer_clause(key, ratings))
    tabular = build_tabular(by_scope, kappa_by_home)
    OUT.mkdir(parents=True, exist_ok=True)
    tex_path = OUT / "analyst_review.tex"
    tex_path.write_text(
        "% requires array, xcolor (table option; load xcolor before arydshln)\n"
        + _tex_fragment(tabular, note, label="tab:analyst_review"),
        encoding="utf-8",
    )
    print(f"TEX  {tex_path}")
    _compile(
        _standalone(tabular, note, page_width="394pt"),
        OUT / "analyst_review.pdf",
        OUT / "analyst_review.png",
    )
    return tex_path


# Question-level funnel. The first five rows partition the 29 questions per home.
# The last three are how many of the agreed-answerable questions also passed later steps.
FUNNEL_ROWS = [
    ("s1_no_ok", "Not answerable, agreed"),
    ("s1_missed", "Missed"),
    ("s1_false_ans", "Wrongly judged answerable"),
    ("s1_unsure", "Step-1 unsure"),
    ("s1_ok", "Agreed answerable"),
    ("s2_ok", r"\ldots\ queries passed"),
    ("s3_ok", r"\ldots\ retrieval passed"),
    ("s4_ok", r"\ldots\ findings passed"),
]
FUNNEL_SCOPES = [*HOMES, "all"]
FUNNEL_NOTE = (
    r"Question-level review funnel, majority vote. "
    r"Each home has 29 questions (87 in all). "
    r"The first five rows partition those questions. "
    r"Missed means the vote said answerable and \ToolName{} said not; "
    r"wrongly judged answerable is the reverse. "
    r"Step-1 unsure is a tied vote among the home's reviewers, and those questions stop. "
    r"The last three rows count how many of the agreed-answerable questions also passed "
    r"the query check, then retrieval, then the findings."
)


def _funnel_nodes(key: pd.DataFrame, consensus: dict) -> dict:
    return _funnel(key, consensus)["nodes"]


def build_funnel_tabular(by_scope: dict) -> list[str]:
    headers = " & ".join(
        [r"\textbf{Outcome}"] + [rf"\textbf{{{s.capitalize()}}}" for s in FUNNEL_SCOPES]
    )
    lines = [
        r"\setlength{\tabcolsep}{8pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        headers + r" \\",
        r"\midrule",
    ]
    for key, label in FUNNEL_ROWS:
        if key == "s1_ok":
            lines.append(r"\midrule")
        cells = [str(int(by_scope[s].get(key, 0))) for s in FUNNEL_SCOPES]
        lines.append(" & ".join([label, *cells]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def write_funnel_table(key: pd.DataFrame | None = None, consensus: dict | None = None) -> Path:
    if key is None or consensus is None:
        key, consensus, _ratings = _load()
    by_scope = {"all": _funnel_nodes(key, consensus)}
    for home in HOMES:
        by_scope[home] = _funnel_nodes(key[key.home == home], consensus)
    tabular = build_funnel_tabular(by_scope)
    OUT.mkdir(parents=True, exist_ok=True)
    tex_path = OUT / "analyst_funnel.tex"
    tex_path.write_text(
        _tex_fragment(tabular, FUNNEL_NOTE, label="tab:analyst_funnel", env="table"),
        encoding="utf-8",
    )
    print(f"TEX  {tex_path}")
    _compile(
        _standalone(tabular, FUNNEL_NOTE, width_cm=12.4),
        OUT / "analyst_funnel.pdf",
        OUT / "analyst_funnel.png",
    )
    return tex_path


# Main-text selection, by question number. The appendix table lists every question.
QUESTION_SELECTION = [4, 15, 9, 11, 21, 20, 26, 27, 28]
QUESTIONS_NOTE = (
    r"Nine of the 29 caregiver questions. "
    r"A check means \ToolName{} treated the question as answerable in that home. "
    r"The last row counts answered questions over all 29, not only the rows shown. "
    r"FULL_LIST"
)
QUESTIONS_ALL_NOTE = (
    r"All 29 caregiver questions. "
    r"A check means \ToolName{} treated the question as answerable in that home. "
    r"The last row counts how many of the 29 questions \ToolName{} answered in each home."
)


def _tex_escape(text: str) -> str:
    return (text.replace("\\", r"\textbackslash{}")
            .replace("&", r"\&").replace("%", r"\%")
            .replace("#", r"\#").replace("_", r"\_"))


def _load_questions() -> list[dict]:
    qfile = json.loads((PROJECT_ROOT / "metadata" / "analyst_questions.json").read_text())
    return sorted(qfile["questions"], key=lambda q: q["num"])


def _analyst_answered() -> dict:
    key = pd.concat(
        [pd.read_csv(p) for p in sorted((RUN / "review" / f"{REVIEW}_key").glob("*_key.csv"))],
        ignore_index=True,
    )
    step1 = key[key.step == 1]
    return {(r.home, r.question_id): _as_bool(r.analyst_answerable) for r in step1.itertuples(index=False)}


def _mark(answered: bool) -> str:
    return r"$\checkmark$" if answered else ""


def _answered_cell(yes: int, n: int) -> str:
    pct = 100.0 * yes / n if n else 0.0
    # \tabularnewline, not \\[dimen]: arydshln redefines \\ and breaks the optional argument.
    return (
        rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}{pct:.1f}\%\tabularnewline"
        rf"{{\scriptsize({yes} out of {n})}}\end{{tabular}}"
    )


def _answered_counts(questions: list[dict], answered: dict) -> list[str]:
    cells = []
    for home in HOMES:
        keys = [f"{q['num']:02d}_{q['id']}" for q in questions]
        yes = sum(1 for key in keys if answered[(home, key)])
        cells.append(_answered_cell(yes, len(keys)))
    return cells


def build_questions_tabular(
    rows: list[dict],
    answered: dict,
    *,
    summary_from: list[dict],
    ellipsis: bool = False,
) -> list[str]:
    grouped: list[tuple[str, list[dict]]] = []
    for q in rows:
        if not grouped or grouped[-1][0] != q["category"]:
            grouped.append((q["category"], []))
        grouped[-1][1].append(q)
    headers = " & ".join(
        ["", r"\textbf{Caregiver's Question}"] + [rf"\textbf{{{h.capitalize()}}}" for h in HOMES]
    )
    lines = [
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        # Question column is whatever the page has left, so the table stays within \linewidth.
        r"\begin{tabular}{@{}c >{\raggedright\arraybackslash}p{\dimexpr\linewidth-7.6cm\relax} ccc@{}}",
        r"\toprule",
        headers + r" \\",
        r"\midrule",
    ]
    for cat, group in grouped:
        span = len(group) + (1 if ellipsis else 0)
        for i, q in enumerate(group):
            key = f"{q['num']:02d}_{q['id']}"
            cat_cell = rf"\multirow{{{span}}}{{*}}{{\rotatebox{{90}}{{{cat}}}}}" if i == 0 else ""
            cells = [_mark(answered[(home, key)]) for home in HOMES]
            lines.append(" & ".join([cat_cell, _tex_escape(q["question"]), *cells]) + r" \\")
        if ellipsis:
            lines.append(r" & \ldots & & & \\")
        if cat != grouped[-1][0]:
            lines.append(r"\midrule")
    lines.append(r"\midrule")
    lines.append(" & ".join(["", r"\textbf{Answered}", *_answered_counts(summary_from, answered)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return lines


def _write_question_table(
    rows: list[dict],
    answered: dict,
    *,
    stem: str,
    label: str,
    note: str,
    standalone_note: str,
    width_cm: float,
    summary_from: list[dict],
    ellipsis: bool = False,
    body_size: str = "",
) -> Path:
    missing = [(home, f"{q['num']:02d}_{q['id']}") for q in summary_from for home in HOMES
               if (home, f"{q['num']:02d}_{q['id']}") not in answered]
    if missing:
        raise KeyError(f"missing step-1 row for {missing[:6]}")
    tabular = build_questions_tabular(
        rows, answered, summary_from=summary_from, ellipsis=ellipsis,
    )
    if body_size:
        tabular.insert(0, body_size)
    OUT.mkdir(parents=True, exist_ok=True)
    tex_path = OUT / f"{stem}.tex"
    fragment = "\n".join([
        r"% requires booktabs, amssymb, array, graphicx, multirow",
        _tex_fragment(tabular, note, label=label, env="table*"),
    ])
    tex_path.write_text(fragment, encoding="utf-8")
    print(f"TEX  {tex_path}")
    _compile(
        _standalone(tabular, standalone_note, width_cm=width_cm),
        OUT / f"{stem}.pdf",
        OUT / f"{stem}.png",
    )
    return tex_path


def write_questions_table(questions: list[dict], answered: dict) -> Path:
    by_num = {q["num"]: q for q in questions}
    rows = [by_num[n] for n in QUESTION_SELECTION]
    full_list = r"The full list of questions is Table~\ref{tab:analyst_questions_all}."
    return _write_question_table(
        rows, answered,
        stem="analyst_questions",
        label="tab:analyst_questions",
        note=QUESTIONS_NOTE.replace("FULL_LIST", full_list),
        standalone_note=QUESTIONS_NOTE.replace("FULL_LIST", "The full list of questions is in the appendix."),
        width_cm=16.5,
        summary_from=questions,
        ellipsis=True,
    )


def write_questions_all_table(questions: list[dict], answered: dict) -> Path:
    return _write_question_table(
        questions, answered,
        stem="analyst_questions_all",
        label="tab:analyst_questions_all",
        note=QUESTIONS_ALL_NOTE,
        standalone_note=QUESTIONS_ALL_NOTE,
        width_cm=16.5,
        summary_from=questions,
        body_size=r"\footnotesize",
    )


if __name__ == "__main__":
    key, consensus, _ratings = _load()
    write_review_table()
    write_funnel_table(key, consensus)
    questions = _load_questions()
    answered = _analyst_answered()
    write_questions_table(questions, answered)
    write_questions_all_table(questions, answered)
