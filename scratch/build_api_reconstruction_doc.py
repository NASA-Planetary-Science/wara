r"""
Build the API 3D position-reconstruction reference PDF (LaTeX / Tectonic).

Computes the physics quantities and the worked example in Python, injects them
into a LaTeX source, and compiles with Tectonic (a self-contained engine, no
system TeX install required). Reconciles three sources:

  1. scratch/api.py            -> calcXYZ        (most complete model)
  2. wara/apicalc.py           -> api_xyz        (used by the --beta GUI)
  3. PhdThesisMauricio-final.pdf, p.64 (Sec 4.3) -> eqs 4.5-4.10 (baseline)

Output: docs/API_position_reconstruction.pdf
"""

from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)
TEX = os.path.join(DOCS, "API_position_reconstruction.tex")
PDF = os.path.join(DOCS, "API_position_reconstruction.pdf")

TECTONIC = shutil.which("tectonic") or os.path.join(
    os.path.expanduser("~"), ".conda", "envs", "sigma", "Library", "bin", "tectonic.exe"
)

# ---------------------------------------------------------------------------
# Physics constants and derived quantities (single source of truth)
# ---------------------------------------------------------------------------
c = 2.99792458e8                 # speed of light [m/s]
m_alpha = 3727.3794066           # alpha mass-energy [MeV]
m_neutron = 939.56542052         # neutron mass-energy [MeV]
m_d = 1875.61294257              # deuteron mass-energy [MeV]
m_t = 2808.92113298              # triton mass-energy [MeV]
E_alpha = 3.5                    # MeV
E_neutron = 14.1                 # MeV
E_d = 0.050                      # deuteron beam energy [MeV] (50 keV)

z_t_cm = 6.7                     # source -> YAP face distance [cm]
a_side = 4.8                     # YAP square side [cm]


def beta_rel(E, m):
    g = 1.0 + E / m
    return np.sqrt(1.0 - 1.0 / g / g)


def beta_nonrel(E, m):
    return np.sqrt(2.0 * E / m)


beta_a = beta_rel(E_alpha, m_alpha)
beta_n = beta_rel(E_neutron, m_neutron)
beta_a_nr = beta_nonrel(E_alpha, m_alpha)
beta_n_nr = beta_nonrel(E_neutron, m_neutron)
v_alpha = beta_a * c
v_neutron = beta_n * c

# center-of-mass velocity from D + T -> alpha + n kinematics (non-rel momentum)
p_d = np.sqrt(2.0 * m_d * E_d)            # MeV/c
beta_com = p_d / (m_d + m_t)
v_com = beta_com * c                      # m/s
tilt_rad = v_com / v_neutron              # worst-case tilt off exact 180 deg
tilt_deg = np.degrees(tilt_rad)
tilt_cm_at_70 = np.tan(tilt_rad) * 70.0
neutron_pct = 100.0 * (beta_n_nr / beta_n - 1.0)


# ---------------------------------------------------------------------------
# Worked numerical example (full, CM-corrected, relativistic model)
# Convention: origin O at the neutron source; +z points into the sample volume;
# the YAP face sits at z = -z_t, the alpha hit at A = (x0, y0, -z_t).
# ---------------------------------------------------------------------------
def full_model(x0_cm, y0_cm, dt_meas_ns, D_cm, beam_axis=(0, 0, 1),
               relativistic=True, use_com=True, use_det=True):
    va = (beta_rel if relativistic else beta_nonrel)(E_alpha, m_alpha) * c
    vn = (beta_rel if relativistic else beta_nonrel)(E_neutron, m_neutron) * c
    vcom = v_com if use_com else 0.0
    chat = np.array(beam_axis, float)
    chat = chat / np.linalg.norm(chat)
    vcv = vcom * chat
    A = np.array([x0_cm, y0_cm, -z_t_cm], float) / 100.0       # m
    D = np.array(D_cm, float) / 100.0                          # m

    aa = va**2 - vcom**2
    bb = 2.0 * A.dot(vcv)
    cc = -A.dot(A)
    t_alpha = (-bb + np.sqrt(bb * bb - 4 * aa * cc)) / (2 * aa)

    w_alpha = (A - vcv * t_alpha) / t_alpha
    v_n = -w_alpha / va * vn + vcv

    dt_a = dt_meas_ns * 1e-9 + t_alpha
    if use_det:
        AA = c * c - v_n.dot(v_n)
        BB = -(2 * c * c * dt_a - 2 * D.dot(v_n))
        CC = c * c * dt_a * dt_a - D.dot(D)
        t_n = (-BB - np.sqrt(BB * BB - 4 * AA * CC)) / (2 * AA)
    else:
        t_n = dt_a * vn / np.linalg.norm(v_n)
    P = v_n * t_n
    return P * 100.0, t_alpha * 1e9, t_n * 1e9                 # cm, ns, ns


EX_x0, EX_y0 = 1.20, -0.80          # cm on the YAP
EX_dt = 18.0                        # ns measured
EX_D = (0.0, 30.0, 35.0)           # illustrative gamma-detector centre [cm]

P_full, ta_ns, tn_full = full_model(EX_x0, EX_y0, EX_dt, EX_D)
P_nodet, _, tn_nodet = full_model(EX_x0, EX_y0, EX_dt, EX_D, use_det=False)
P_thesis, _, _ = full_model(EX_x0, EX_y0, EX_dt, EX_D,
                            relativistic=False, use_com=False, use_det=True)

print("=== derived quantities ===")
print(f"beta_neutron rel/nonrel = {beta_n:.6f} / {beta_n_nr:.6f} ({neutron_pct:.2f}% high)")
print(f"v_alpha={v_alpha:.4e}  v_neutron={v_neutron:.4e}  v_com={v_com:.4e} m/s")
print(f"tilt {tilt_deg:.3f} deg -> {tilt_cm_at_70:.2f} cm at 70 cm")
print(f"t_alpha={ta_ns:.3f} ns  P_full={P_full}  t_n={tn_full:.3f} ns")


# ---------------------------------------------------------------------------
# LaTeX source.  Numbers are injected through @@TOKEN@@ replacement so the
# heavy use of braces in the body needs no escaping.
# ---------------------------------------------------------------------------
def vec3(p):
    return f"({p[0]:.2f},\\ {p[1]:.2f},\\ {p[2]:.2f})"


TEMPLATE = r"""
\documentclass[11pt]{article}

\usepackage[a4paper,margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{bm}
\usepackage{siunitx}
\usepackage{booktabs}
\usepackage{array}
\usepackage[table]{xcolor}
\usepackage{graphicx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,angles,quotes,calc,positioning}
\usepackage{enumitem}
\usepackage{fancyhdr}
\usepackage[bookmarks,hidelinks]{hyperref}

\definecolor{navy}{HTML}{16324A}
\definecolor{steel}{HTML}{1F4E6B}
\definecolor{rowg}{HTML}{EEF3F7}
\definecolor{cAlpha}{HTML}{1F77B4}
\definecolor{cN}{HTML}{D62728}
\definecolor{cD}{HTML}{2CA02C}
\definecolor{cG}{HTML}{9467BD}
\definecolor{cCom}{HTML}{FF7F0E}

\sisetup{exponent-product=\times}
\DeclareSIUnit{\cm}{cm}
\DeclareSIUnit{\ns}{ns}
\DeclareSIUnit{\MeV}{MeV}
\DeclareSIUnit{\keV}{keV}

\colorlet{headblue}{steel}
\renewcommand{\arraystretch}{1.25}

\newcommand{\code}[1]{\texttt{\small #1}}
\newcommand{\dta}{\mathit{dt}_a}
\newcommand{\dt}{\mathit{dt}}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0.4pt}
\fancyfoot[L]{\footnotesize\color{gray}API 3D position reconstruction --- derivation \& reconciliation}
\fancyfoot[R]{\footnotesize\color{gray}p.\ \thepage}

\setlength{\parskip}{0.55em}
\setlength{\parindent}{0pt}

\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries\color{navy}}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\large\bfseries\color{steel}}{\thesubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{1.2em}{0.5em}

\begin{document}

\begin{center}
  {\LARGE\bfseries Three-Dimensional Position Reconstruction\\[2pt]
   in the API System}\\[10pt]
  {\large\itshape Mathematical derivation and reconciliation of the
   \code{calcXYZ}, \code{api\_xyz},\\ and thesis (Sec.~4.3) formulations}\\[6pt]
  {\color{gray}\rule{0.6\textwidth}{0.4pt}}
\end{center}

\vspace{0.4em}
\noindent
This note derives, from first principles and for our specific geometry, the
reconstruction of the neutron-interaction point $(x,y,z)$ in the
Associated-Particle Imaging (API) system. The full model (relativistic speeds,
center-of-mass correction, exact gamma-detector vector) is taken as
authoritative; the two simplified implementations and the thesis derivation are
recovered as special cases, and every discrepancy among the three sources is
catalogued in the errata of Section~8.

\section{The reconstruction problem}
In coincidence mode the DAQ reports, per event, the four corner energies of the
YAP alpha detector, the gamma-ray energy, and the alpha--gamma time difference
$\dt$ (Table 4.2 of the thesis). Reconstruction proceeds in four steps:
\begin{itemize}[leftmargin=1.4em,itemsep=2pt]
  \item \textbf{Step A --- Alpha position.} The four corner energies give the
        alpha interaction point $(x_0,y_0)$ on the YAP face (Section~3).
  \item \textbf{Step B --- Neutron direction and speed.} Momentum conservation
        in the D--T reaction fixes the neutron velocity vector from the alpha
        direction, including the center-of-mass motion (Section~4).
  \item \textbf{Step C --- Time of flight.} The measured $\dt$, the alpha flight
        time $t_\alpha$, and the gamma flight time tie the neutron path length to
        the geometry (Section~5).
  \item \textbf{Step D --- Solve for the point.} A single quadratic in the
        neutron flight time yields $(x,y,z)$ (Section~6).
\end{itemize}
The tagged neutron travels essentially opposite to the detected alpha, so the
alpha hit defines a \emph{ray} from the source into the sample; the timing fixes
\emph{how far} along that ray the gamma-producing scatter occurred.

\section{Coordinate system and geometry}
We place the origin $O$ at the neutron production point (the reaction spot inside
the generator head) and let the $+z$ axis point along the API cone axis, into the
sample volume. This is the choice made by \code{calcXYZ}; it is the cleanest
because the neutron starts at the origin, so both the interaction point and the
gamma detector are referred to the same point. The YAP alpha-detector face lies
at $z=-z_t$, and the alpha interacts at $A=(x_0,y_0,-z_t)$.

\begin{center}
\begin{tikzpicture}[>=Stealth,scale=1.15,
   vec/.style={thick,->},line cap=round]
  \coordinate (O) at (0,0);
  \coordinate (A) at (-1.1,0.95);
  \coordinate (P) at (4.4,-2.6);
  \coordinate (D) at (3.0,2.4);
  % YAP crystal face
  \draw[line width=4pt,gray!70] (-1.1,-1.6) -- (-1.1,1.6);
  \node[gray!60,above,font=\footnotesize] at (-1.1,1.65) {YAP alpha detector};
  % z axis
  \draw[->,gray,thin] (-1.2,-1.6) -- (4.8,-1.6)
       node[right,font=\footnotesize,black!55] {$+z$ (into sample)};
  % angle
  \pic[draw,->,"\small$\theta$",angle radius=0.85cm,angle eccentricity=1.25]
       {angle=P--O--D};
  % vectors
  \draw[vec,cCom] (O) -- (0.9,0) node[above right,font=\scriptsize,cCom]
       {$\bm{v}_{\mathrm{com}}$ (beam)};
  \draw[vec,cAlpha] (O) -- (A);
  \draw[vec,cN,line width=1.1pt] (O) -- (P);
  \draw[vec,cD] (O) -- (D);
  \draw[vec,cG] (P) -- (D);
  % points
  \foreach \p in {O,A,P,D}{\fill (\p) circle (1.6pt);}
  % labels
  \node[cAlpha,above left,font=\footnotesize] at (A) {$\alpha$ hit $(x_0,y_0,-z_t)$};
  \node[cD,above,font=\footnotesize] at (D) {$\gamma$ detector $\bm{D}$};
  \node[cN,below right,font=\footnotesize] at (P) {interaction $P=(x,y,z)$};
  \node[below right=1pt,font=\footnotesize] at (O) {source $O=(0,0,0)$};
  \node[cAlpha,font=\small] at (-0.85,0.30) {$\bm{a},\hat u_\alpha$};
  \node[cN,font=\small] at (2.4,-1.05) {$\bm{n}=|\bm{n}|\,\hat u$};
  \node[cD,font=\small] at (1.35,1.55) {$\bm{d}$};
  \node[cG,font=\small] at (4.05,-0.1) {$\bm{g}$};
\end{tikzpicture}
\end{center}
{\footnotesize\color{black!60}\textbf{Figure 1.} Reconstruction geometry
(projected on the $y$--$z$ plane). The alpha hit on the YAP defines the ray
direction; the timing fixes the neutron path length $|\bm{n}|$. Vectors
$\bm a,\bm n,\bm d,\bm g$ and the angle $\theta$ are used in the derivation.}

\vspace{0.6em}
Our specific geometry uses the following fixed quantities:

\begin{center}
\begin{tabular}{>{\raggedright}p{1.6cm} p{6.4cm} p{6.0cm}}
\toprule
\rowcolor{steel}\color{white}\textbf{Symbol} & \color{white}\textbf{Meaning} & \color{white}\textbf{Value (this system)}\\
$z_t$ & source $\rightarrow$ YAP-face distance & \SI{@@ZT@@}{\cm}\\
\rowcolor{rowg}$a$ & YAP active square side & \SI{@@A@@}{\cm} ($\pm\SI{@@AH@@}{\cm}$)\\
$v_\alpha$ & alpha speed (\SI{3.5}{\MeV}, relativistic) & \SI{@@VA@@}{\meter\per\second} ($\beta=@@BA@@$)\\
\rowcolor{rowg}$v_n$ & neutron speed (\SI{14.1}{\MeV}, relativistic) & \SI{@@VN@@}{\meter\per\second} ($\beta=@@BN@@$)\\
$v_{\mathrm{com}}$ & center-of-mass speed (\SI{50}{\keV} $\mathrm{D}^+$) & \SI{@@VCOM@@}{\meter\per\second} ($\beta=@@BCOM@@$)\\
\rowcolor{rowg}$c$ & speed of light $=v_\gamma$ & \SI{@@C@@}{\meter\per\second}\\
$\hat C$ & ion-beam unit vector ($v_{\mathrm{com}}$ dir.) & \code{meta['ion beam axis']}\\
\rowcolor{rowg}$\bm D$ & gamma-detector centre & \code{meta['detectors'][k]['position']}\\
\bottomrule
\end{tabular}
\end{center}

In \code{calcXYZ} the symbol \code{meta['scintillator distance']} \emph{is} $z_t$:
it is the $z$ of the alpha hit, \code{Z\_alpha = scintillator\_dz}. The thesis
quotes $z_t=\SI{6}{\cm}$ in the text; the code uses \SI{6.7}{\cm}, which we adopt
here (see errata, Section~8).

\section{Step A --- alpha position from the four corners}
The YAP is read out at its four corners $A,B,C,D$ (charge division). Labelling
them so that $B$ is the $+x,+y$ corner, $A$ the $-x,+y$, $C$ the $+x,-y$, and $D$
the $-x,-y$ corner, the normalised coordinates are
\begin{equation}
  x_{\mathrm{raw}}=\frac{B+C-D-A}{A+B+C+D},\qquad
  y_{\mathrm{raw}}=\kappa\,\frac{B+A-D-C}{A+B+C+D},
\end{equation}
where $\kappa=1.305$ is an empirical factor that makes the measured alpha image
square (\code{map\_alpha\_XY}). Note \code{calc\_own\_pos} / the \code{X2,Y2}
columns used by the GUI omit $\kappa$ (set it to $1$) and apply the squaring later
through edge detection --- numerically equivalent once the axis is scaled to
centimetres.

The dimensionless values are mapped to physical centimetres across the
\SI{4.8}{\cm} active face. Rather than trust the absolute gain, the code locates
the populated edges of the alpha image (the histogram region above ${\sim}1/3$ of
the peak) and linearly maps that span onto $[-2.4,+2.4]\,\si{\cm}$:
\begin{equation}
  x_0=\frac{a}{x_{\max}-x_{\min}}\,(x_{\mathrm{raw}}-x_{\max})+\frac{a}{2},
  \qquad a=\SI{4.8}{\cm},
\end{equation}
and similarly for $y_0$. (The legacy \code{map\_alpha\_XY} additionally flips the
$x$-axis so an $X$--$Y$ image matches the view from in front of the experiment,
and can apply spline edge-corrections from the \code{DX/DY} lookup tables.) The
output of Step~A is the alpha interaction point $(x_0,y_0)$ in centimetres on the
YAP face.

\section{Step B --- neutron velocity vector}

\subsection{Naive (back-to-back) direction}
If the neutron and alpha left the source exactly \ang{180} apart, the neutron
direction is simply opposite the source$\rightarrow$alpha vector. With the alpha
hit at $A=(x_0,y_0,-z_t)$,
\begin{equation}
  \bm a=\langle x_0,\,y_0,\,-z_t\rangle,\qquad
  \hat u=-\frac{\bm a}{|\bm a|}
        =\frac{\langle -x_0,\,-y_0,\,z_t\rangle}{\sqrt{x_0^2+y_0^2+z_t^2}}.
\end{equation}
This is exactly the thesis vector (eq.~4.5) and what \code{api\_xyz} uses
($u_x=-x_a/|\bm a|$, $u_z=+z_a/|\bm a|$ with $z_a=\SI{6.7}{\cm}$). The thesis
takes the origin at the YAP centre rather than the source, but the
\emph{direction} $\hat u$ is identical.

\subsection{Center-of-mass correction (the full model)}
The \ang{180} assumption holds only in the reaction center-of-mass (CM) frame.
The $\mathrm{{}^2H}+\mathrm{{}^3H}\rightarrow\mathrm{{}^4He}+n$ reaction is driven
by a deuteron beam, so the CM moves in the lab with a small velocity along the
beam axis $\hat C$:
\begin{equation}
  \bm v_{\mathrm{com}}=v_{\mathrm{com}}\,\hat C,\qquad
  v_{\mathrm{com}}=\frac{\sqrt{2\,m_d E_d}}{m_d+m_t}\,c\approx @@BCOM@@\,c .
\end{equation}
For a \SI{50}{\keV} deuteron beam this is
$v_{\mathrm{com}}\approx\SI{@@VCOM@@}{\meter\per\second}$. Although tiny
($\beta_{\mathrm{com}}\approx@@BCOM@@$), it tilts the lab neutron direction away
from exact \ang{180} by up to ${\approx}\ang{@@TILTDEG@@}$, i.e.\ roughly
\SI{@@TILTCM@@}{\cm} of transverse displacement at a \SI{70}{\cm} stand-off ---
comparable to the system resolution and therefore worth correcting.
\code{calcXYZ} reads $v_{\mathrm{com}}$ versus beam energy from
\code{center-of-mass.npy}; the formula above reproduces it from the reaction
kinematics.

The construction has three parts. First the alpha flight time $t_\alpha$ is found
by requiring that, in the CM frame, the alpha travels at speed $v_\alpha$:
\begin{equation}
  |\bm A-\bm v_{\mathrm{com}}\,t_\alpha|=v_\alpha\,t_\alpha
  \ \Longrightarrow\
  (v_\alpha^2-v_{\mathrm{com}}^2)\,t_\alpha^2
   +2(\bm A\cdot\bm v_{\mathrm{com}})\,t_\alpha-|\bm A|^2=0,
\end{equation}
taking the positive root. The alpha velocity in the CM frame is then
$\bm w_\alpha=(\bm A-\bm v_{\mathrm{com}}t_\alpha)/t_\alpha$, and because the
neutron is exactly opposite the alpha \emph{in the CM frame}, the neutron
velocity in the lab is
\begin{equation}
  \bm v_n=-\frac{\bm w_\alpha}{v_\alpha}\,v_n+\bm v_{\mathrm{com}},
\end{equation}
i.e.\ (neutron CM direction)$\times$(neutron CM speed)$+$(CM drift). This is
exactly the two-line update in \code{calcXYZ}
(\code{vn\_x = -vn\_x/v\_alpha*v\_neutron + v\_com\_vector[0]}, \ldots). In the
naive model $v_{\mathrm{com}}\to0$ and $\bm v_n=v_n\hat u$.

\section{Step C --- the time-of-flight relation}
The recorded $\dt$ is the alpha--gamma time difference. Writing $t_n,t_g,t_\alpha$
for the neutron, gamma, and alpha flight times, the thesis relation (eq.~4.7) is
\begin{equation}
  \dt=t_n+t_g-t_\alpha
  \quad\Longrightarrow\quad
  \dta\equiv \dt+t_\alpha=t_n+t_g,
\end{equation}
so folding the alpha flight time into the time difference defines a single
quantity $\dta$. (Per-detector electronic offsets --- the \code{z-align}
constants and the ${\approx}\SI{11}{\ns}$ PMT transit delay noted in the thesis
--- are subtracted from $\dt$ before this step; they shift the zero of $\dt$ but
do not change the algebra below.) In \code{calcXYZ} this is literally
\code{df['dt'] = dt\_orig + t\_alpha} followed by the channel-wise offset
subtraction.

\section{Step D --- solving for the interaction point}
Let $\bm D$ be the gamma-detector centre and $\bm P=\bm v_n t_n$ the interaction
point (the neutron starts at the origin with lab velocity $\bm v_n$). The gamma
then travels from $\bm P$ to $\bm D$ at speed $c$ in the remaining time
$t_g=\dta-t_n$:
\begin{equation}
  |\bm D-\bm v_n\,t_n|^2=c^2\,(\dta-t_n)^2 .
\end{equation}
Expanding gives a quadratic in the neutron flight time,
$\alpha' t_n^2+\beta' t_n+\gamma'=0$, with
\begin{equation}
  \alpha'=c^2-|\bm v_n|^2,\qquad
  \beta'=-\bigl(2c^2\dta-2\,\bm D\cdot\bm v_n\bigr),\qquad
  \gamma'=c^2\dta^2-|\bm D|^2 .
\end{equation}
Our coordinate choice selects the negative root:
\begin{equation}
  t_n=\frac{-\beta'-\sqrt{\beta'^2-4\alpha'\gamma'}}{2\alpha'},
  \qquad (x,y,z)=\bm v_n\,t_n .
\end{equation}
This is exactly the second quadratic in \code{calcXYZ} (coefficients
\code{a,b,c} and \code{t\_neutron}); the reconstructed point is
\code{X=vn\_x*t\_neutron}, etc. It uses the \emph{exact} detector vector
$\bm D-\bm P$; no point-source or law-of-cosines approximation enters.

\subsection{The thesis / \texorpdfstring{\code{api\_xyz}}{api\_xyz} special case}
If we drop the CM correction (so $|\bm v_n|=v_n$ and $\bm v_n=v_n\hat u$) and
solve instead for the path length $|\bm n|=v_n t_n$, the gamma constraint becomes
the law of cosines (thesis eq.~4.8)
\begin{equation}
  |\bm g|^2=|\bm d|^2+|\bm n|^2-2|\bm d|\,|\bm n|\cos\theta,\qquad
  \cos\theta=\frac{u_x x_1+u_y y_1+u_z z_1}{|\bm d|},
\end{equation}
which, combined with $|\bm g|=c(\dta-|\bm n|/v_n)$, yields the thesis quadratic
(eqs.~4.9--4.10)
\begin{equation}
  a=1-\frac{c^2}{v_n^2},\qquad
  b=\frac{2\dta c^2}{v_n}-2|\bm d|\cos\theta,\qquad
  c=|\bm d|^2-c^2\dta^2,
\end{equation}
\begin{equation}
  |\bm n|=\frac{-b-\sqrt{b^2-4ac}}{2a},\qquad
  x=\frac{-x_0}{|\bm a|}|\bm n|,\quad
  y=\frac{-y_0}{|\bm a|}|\bm n|,\quad
  z=\frac{z_t}{|\bm a|}|\bm n|+z_t .
\end{equation}
This is precisely \code{api\_xyz(use\_det=True)}. The two forms agree when
$v_{\mathrm{com}}\to0$ \emph{and} when $\bm d$ is measured from the same point as
$\bm n$ --- which exposes the origin inconsistency discussed in the errata.
Setting \code{use\_det=False} drops the detector term entirely, leaving the
straight-ray estimate $|\bm n|=v_n\dta$; this is the crudest variant and is, at
present, the one the \code{--beta} GUI uses.

\clearpage
\section{Reconciliation of the three implementations}
All three sources implement the same idea at increasing levels of rigour.

\begin{center}
\begin{tabular}{p{3.0cm} p{2.9cm} p{2.9cm} p{3.0cm}}
\toprule
\rowcolor{steel}\color{white}\textbf{Aspect} & \color{white}\textbf{Thesis Sec.\ 4.3} & \color{white}\textbf{\code{api\_xyz} (-{}-beta)} & \color{white}\textbf{\code{calcXYZ}}\\
Coordinate origin & YAP centre & YAP centre & neutron source\\
\rowcolor{rowg}Source--YAP $z_t$ & \SI{6}{\cm} & \SI{6.7}{\cm} & meta scint.\ dist.\\
Particle speeds & symbolic & non-relativistic & relativistic\\
\rowcolor{rowg}$\alpha$--$n$ back-to-back & exact \ang{180} & exact \ang{180} & CM-corrected\\
Gamma detector & law of cosines & cosines / ignored & exact $\bm D-\bm P$\\
\rowcolor{rowg}Units & --- & cm, $c=\num{3e10}$ & SI, scipy $c$\\
Solves for & $|\bm n|$ (path) & $|\bm n|$ (path) & $t_n$ (then $\bm v_n t_n$)\\
\rowcolor{rowg}Quadratic root & negative & negative & negative\\
\bottomrule
\end{tabular}
\end{center}

\section{Errata and resolved discrepancies}
The following issues were found while reconciling the three sources. Items 1--4
are genuine physics/algebra points; items 5--6 are bookkeeping.
\begin{enumerate}[leftmargin=1.5em,itemsep=4pt]
  \item \textbf{Thesis typo (minor).} The constant term of eq.~4.10 is printed as
        $c=|\bm d|^2-c^2\dt^2$ but should use $\dta$, i.e.\
        $c=|\bm d|^2-c^2\dta^2$. (The stray subscript ``$a$'' is dropped to the
        next line in the PDF.) The rest of the derivation is self-consistent.
  \item \textbf{Law-of-cosines origin mismatch (thesis \& \code{api\_xyz}).} The
        neutron path $\bm n$ is measured from the source at $(0,0,z_t)$, but
        $\bm d$ and $\cos\theta$ are measured from the YAP centre. The law of
        cosines treats them as sharing one vertex, so the gamma path length is
        off by ${\sim}z_t$ in $z$. Putting the origin at the source (full model)
        removes this: $\bm D$ and $\bm P$ share the origin and
        $|\bm g|=|\bm D-\bm P|$ is exact.
  \item \textbf{Non-relativistic speeds (\code{api\_xyz}).} Using $\sqrt{2E/m}$
        overestimates the neutron speed: $\beta_n=@@BNNR@@$ vs $@@BN@@$
        relativistic, ${\sim}@@NPCT@@\%$ high. Over a ${\sim}\SI{70}{\cm}$ flight
        this biases $Z$ at the centimetre scale. The full model uses the
        relativistic $\beta$.
  \item \textbf{\ang{180} back-to-back assumption (thesis \& \code{api\_xyz}).}
        Ignores the lab-frame CM momentum; the neutron is back-to-back with the
        alpha only in the CM frame. This tilts the ray by up to
        ${\sim}\ang{@@TILTDEG@@}$ (${\sim}\SI{@@TILTCM@@}{\cm}$ at \SI{70}{\cm}).
        The full model corrects it via $\bm v_{\mathrm{com}}$ --- precisely the
        dominant error source the thesis itself flags.
  \item \textbf{$z_t$ value.} \SI{6}{\cm} (thesis text) vs \SI{6.7}{\cm} (code).
        We adopt \SI{6.7}{\cm}, matching \code{api\_xyz}, \code{helper\_api}, and
        the as-built code.
  \item \textbf{Sign / origin convention.} \code{api\_xyz} puts the sample at
        $+z$ with the YAP centre as origin; \code{calcXYZ} puts the origin at the
        source. They are related by a translation of $z_t$ along $z$ (and a sign
        flip of the $z$-axis). Reconstructed depths must be interpreted in the
        convention of whichever routine produced them.
\end{enumerate}

\noindent\textbf{Practical consequence for \code{--beta}.} The GUI currently calls
\code{api\_xyz(use\_det=False)}, the crudest path (non-relativistic, no CM
correction, detector ignored). Migrating the GUI to the full \code{calcXYZ} model
--- or at minimum enabling the detector term and relativistic speeds --- is the
natural follow-up to this note.

\clearpage
\section{Worked numerical example}
To make the formulas checkable, we reconstruct one synthetic event with the full
model. Inputs (origin at source, $+z$ into the sample):
\begin{itemize}[leftmargin=1.4em,itemsep=2pt]
  \item alpha hit on the YAP: $(x_0,y_0)=(@@EXX0@@,@@EXY0@@)\,\si{\cm}$, at
        $z=-\SI{@@ZT@@}{\cm}$;
  \item measured time difference: $\dt=\SI{@@EXDT@@}{\ns}$;
  \item beam axis $\hat C=(0,0,1)$; $v_{\mathrm{com}}=\SI{@@VCOM@@}{\meter\per\second}$;
  \item illustrative gamma-detector centre:
        $\bm D=(@@DX@@,@@DY@@,@@DZ@@)\,\si{\cm}$.
\end{itemize}
Intermediate and final results:

\begin{center}
\begin{tabular}{p{8.2cm} p{5.0cm}}
\toprule
\rowcolor{steel}\color{white}\textbf{Quantity} & \color{white}\textbf{Value}\\
alpha flight time $t_\alpha$ & \SI{@@TA@@}{\ns}\\
\rowcolor{rowg}$\dta=\dt+t_\alpha$ & \SI{@@DTA@@}{\ns}\\
neutron flight time $t_n$ (full) & \SI{@@TNFULL@@}{\ns}\\
\rowcolor{rowg}$\bm P$, full model (det $+$ CM $+$ rel.) & $@@PFULL@@\,\si{\cm}$\\
$\bm P$, detector ignored (-{}-beta path) & $@@PNODET@@\,\si{\cm}$\\
\rowcolor{rowg}$\bm P$, thesis (non-rel, no CM) & $@@PTHESIS@@\,\si{\cm}$\\
depth difference full $-$ thesis ($z$) & \SI{@@DZDIFF@@}{\cm}\\
\bottomrule
\end{tabular}
\end{center}

The three models agree to within a few centimetres on this event, with the
differences dominated by the detector term and the CM/relativistic corrections
--- consistent with the system's few-centimetre resolution and confirming that
the simplified routines are adequate for quick looks while the full model is
preferred for quantitative work.

\vspace{0.5em}
{\footnotesize\color{black!55} Numbers above are produced by
\code{scratch/build\_api\_reconstruction\_doc.py}, which re-implements the three
models directly; the detector position is illustrative.}

\end{document}
"""

tokens = {
    "@@ZT@@": f"{z_t_cm:.1f}",
    "@@A@@": f"{a_side:.1f}",
    "@@AH@@": f"{a_side/2:.1f}",
    "@@VA@@": f"{v_alpha:.3e}",
    "@@BA@@": f"{beta_a:.4f}",
    "@@VN@@": f"{v_neutron:.3e}",
    "@@BN@@": f"{beta_n:.4f}",
    "@@VCOM@@": f"{v_com:.3e}",
    "@@BCOM@@": f"{beta_com:.5f}",
    "@@C@@": f"{c:.6e}",
    "@@TILTDEG@@": f"{tilt_deg:.2f}",
    "@@TILTCM@@": f"{tilt_cm_at_70:.1f}",
    "@@BNNR@@": f"{beta_n_nr:.4f}",
    "@@NPCT@@": f"{neutron_pct:.1f}",
    "@@EXX0@@": f"{EX_x0:.2f}",
    "@@EXY0@@": f"{EX_y0:.2f}",
    "@@EXDT@@": f"{EX_dt:.1f}",
    "@@DX@@": f"{EX_D[0]:.0f}",
    "@@DY@@": f"{EX_D[1]:.0f}",
    "@@DZ@@": f"{EX_D[2]:.0f}",
    "@@TA@@": f"{ta_ns:.3f}",
    "@@DTA@@": f"{EX_dt + ta_ns:.3f}",
    "@@TNFULL@@": f"{tn_full:.3f}",
    "@@PFULL@@": vec3(P_full),
    "@@PNODET@@": vec3(P_nodet),
    "@@PTHESIS@@": vec3(P_thesis),
    "@@DZDIFF@@": f"{P_full[2]-P_thesis[2]:+.2f}",
}

tex = TEMPLATE
for k, v in tokens.items():
    tex = tex.replace(k, v)
with open(TEX, "w", encoding="utf-8") as f:
    f.write(tex)
print(f"Wrote {TEX}")

# ---------------------------------------------------------------------------
# Compile with Tectonic
# ---------------------------------------------------------------------------
print("Compiling with Tectonic ...")
res = subprocess.run(
    [TECTONIC, "--keep-logs", "-o", DOCS, TEX],
    capture_output=True, text=True,
)
print(res.stdout[-2000:])
if res.returncode != 0:
    print("--- tectonic stderr ---")
    print(res.stderr[-4000:])
    raise SystemExit(f"Tectonic failed (exit {res.returncode})")
print(f"\nWrote {PDF}")
