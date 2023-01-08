+++
title="TikZ Externalization for Beamer"
description="Compile a single file if the figure doesn't change across overlays"
date=2021-10-27

[taxonomies]
categories = ["Research"]
tags = ["latex", "tikz", "beamer"]

[extra]
toc = true
+++

Recently I made a presentation for a talk on [FWCG2021](https://comptag.github.io/fwcg21/). As always I went to Beamer. What was different this time was that I needed many figures, I mean, a lot. On top of that, I wanted to do overlays on these figures. Overleaf timed out when trying to compile all the TikZ figures so I turned to [`external`](https://www.overleaf.com/learn/latex/Questions/I_have_a_lot_of_tikz%2C_matlab2tikz_or_pgfplots_figures%2C_so_I%27m_getting_a_compilation_timeout._Can_I_externalise_my_figures%3F). There are a lot of files to generate when I built the presentation locally and it was unbearably slow. Maybe there's a way to reduce the time, at least a little bit?

<!-- more -->

# Vanilla TikZ Externalization
Using the vanilla TikZ externalization is easy, simply put
```ltx
\usetikzlibrary{external}
\tikzexternalize % or optionally \tikzexternalize[prefix=some_directory/]
```
in the `main.tex`. Now the LaTeX compiler will spawn another process to compile a TikZ environment into some separated files whenever it encounters one.

Properties of the vanilla approach:
- By default, the name of the files generating by an environment will be `figure-{num}`, where `{num}` is automatically incremented.
  Using `\tikzsetnextfilename` could set the file name, but the "same" environment in later overlays will not be generated since they have the same name and TikZ doesn't think they are different.
- Each TikZ environment generates 4 files:
  1. `{name}.log`: the log of the individual process compiling this environment.
  2. `{name}.md5`: the MD5 hash of the input to the TikZ environment, pre-expaned.
  3. `{name}.dpth`: the auxiliary file to include when loading externalized figure.
  4. `{name}.pdf`: the generated figure.

So if we set up a TikZ figure as follows,
```ltx
\begin{frame}
  \begin{tikzpicture}
    \node[draw, circle] (n) at (0, 0) {\alt<2>{1}{2}};
  \end{tikzpicture}
\end{frame}
```
where we want the node label to be `1` on the first overlay, and `2` on the second overlay. In this snippet, beamer won't recognize the existence of the second overlay, and only one overlay will be generated.

# Prior Arts
The most immediate problem to solve is to make externalization to generate different figures across overlays. Such question was asked and answered before.

## Without Custom Filename
The earliest reference I could find is due to [this TeX stackexchange question](https://tex.stackexchange.com/questions/78955/use-tikz-external-feature-with-beamer-only), in which 2 methods were given. The main idea is to let beamer know how many overlay is needed, and externalization should automatically generate figures for each overlay, with incrementing name.

- [Christian Feuersänger's Solution](https://tex.stackexchange.com/a/79461/73509)

  The idea is simple, wrap the TikZ environment by a `\only` call indicating the number of overlays needed in the frame. So the prior example becomes:
  ```ltx
  \begin{frame}
    \only<1-2>{%
      \begin{tikzpicture}
        \node[draw, circle] (n) at (0, 0) {\alt<2>{1}{2}};
      \end{tikzpicture}%
    }%
  \end{frame}
  ```
  One downside is the need to explicitly spell out the required overlays.
- [Andrew Stacey's Solution](https://tex.stackexchange.com/a/79572/73509)

  The idea is to communicate the need for more overlays in the `.dpth` file to the main process, and turn off some optimizations to make sure such communication happens. Andrew made a TikZ preset for this:
  ```ltx
  \makeatletter
  \tikzset{
    beamer externalising/.style={%
      execute at end picture={%
        \tikzifexternalizing{%
          \ifbeamer@anotherslide
          \pgfexternalstorecommand{\string\global\string\beamer@anotherslidetrue}%
          \fi
        }{}%
      }%
    },
    external/optimize=false
  }
  \makeatother
  ```
  And all it takes to use it is just to apply the style for the environment:
  ```ltx
  \begin{frame}
    \begin{tikzpicture}[beamer externalising]
      \node[draw, circle] (n) at (0, 0) {\alt<2>{1}{2}};
    \end{tikzpicture}
  \end{frame}
  ```

The main drawbacks of these two solutions are:
- Setting a custom filename for the figure does not work. Consider the following snippet:
  ```ltx
  \begin{frame}
    \tikzsetnextfilename{test}
    \only<1-2>{%
      \begin{tikzpicture}
        \node[draw, circle] (n) at (0, 0) {\alt<2>{1}{2}};
      \end{tikzpicture}%
    }%
  \end{frame}
  ```
  1. On overlay 1, `test.log`, `test.md5`, `test.dpth`, and `test.pdf` are produced, which correctly reflect the first overlay.
  2. On overlay 2, the **unexpanded** but canonicalized body of `tikzpicture` is passed for MD5 digest, which is obviously identical to the previous overlay. The externalization process finds `test.md5` and deems that it is fine to just use that.

  The result is an unchanged figure despite a `\alt` command in it.
- Suppose in a 3-overlay frame the figure only changes in overlay 2, the externalization will still generate 3 figures instead of 2.
## With Custom Filename
Andrew Stacey addressed the first drawback in [this question](https://tex.stackexchange.com/a/119440/73509). The idea is simply to attach the overlay number to the name when using `\tikzsetnextfilename`, so that figures in different overlays get different filenames. It is achieved by simply redefining `\tikzsetnextfilename` as:
```ltx
\let\orig@tikzsetnextfilename=\tikzsetnextfilename
\renewcommand\tikzsetnextfilename[1]{\orig@tikzsetnextfilename{#1-\overlaynumber}}
```

The second drawback is still present, and I came up with a partial solution for that.
# Introducing `\tikznamebeamer`
The idea is to have user to specify different groups of overlay numbers, and generate one figure per group. Here is the implementation:
```ltx
\usepackage{etoolbox}
\makeatletter
\newcounter{tikznamebeamer@tempcnt}% counter variable to go through groups
\def\tikznamebeamer@finalidx{}% will store the group index to use
\newcommand{\tikznamebeamer}[2][figure]{%
    \def\tikznamebeamer@slidesetidx{%
        \setcounter{tikznamebeamer@tempcnt}{0}% reset counter
        \let\tikznamebeamer@finalidx=0% default group being 0
        \renewcommand*{\do}[1]{% loop
            \stepcounter{tikznamebeamer@tempcnt}% counter++
            \let\tikznamebeamer@templist=\relax%
            \forcsvlist{\listadd\tikznamebeamer@templist}{####1}% iterate through list of overlay numbers
            \xifinlist{\number\beamer@slideinframe}{\tikznamebeamer@templist}{%
                \edef\tikznamebeamer@finalidx{\thetikznamebeamer@tempcnt}% found, set group number
            }{}%
        }%
        \docsvlist{#2}% iterate through a list of lists of overlay numbers
    }%
    \tikznamebeamer@slidesetidx%
    \tikzsetnextfilename{#1-\tikznamebeamer@finalidx}%
}
\makeatother
```
`\tikznamebeamer` accepts two arguments. The first argument is the desired filename. The second argument is a list of lists of numbers. For example, suppose the current frame has 5 overlays, `{{2, 5}, 4}` defines **3** groups (note that any overlay number of specified goes to group 0):
- group 0 (default): 1, 3
- gruop 1: 2, 5
- group 2: 4

Hence a call `\tikznamebeamer[my-graph]{{2, 5}, 4}` in this case will create 3 sets of files with name `my-graph-0`, `my-graph-1`, and `my-graph-2`. Overlay will use the correct set of files according to its group number.

In this way, we can both specify a custom filename, and avoid unnecessary compilation at the same time.

# Future Directions
It is quite tedious to manually specify what overlay numbers have the same figure. Ideally the externalization should:
1. Hash the canonicalized body of `tikzpicture` **after expanding beamer related macros**.
2. Match the hash within the generated files to decide whether to create a new one or not.

I haven't figured out the expanding part yet. But here are some thoughts on the second part:
- One simple idea is to append the hash to the end of the filename. Downside is that the figure could be frequently edited, and old files quickly bloat the build directory.
- Another more practical idea is to store a mapping between hash and filename in memory. This seems  much more complicated to implement and though.

Of course the elephant in the room is that the compilation is single-threaded, where the TikZ compilation chould've been done in parallel. There are some techniques involving using a custom build system for figures but that does not play too well on Overleaf.