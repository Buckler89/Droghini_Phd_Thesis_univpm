%%
%% This is file `univpmphdthesis.cls',
%% generated with the docstrip utility.
%%
%% The original source files were:
%%
%% univpmphdthesis.dtx  (with options: `class')
%% _______________________________________________________________
%% 
%% The univpmphdthesis class for typesetting the PhD
%% thesis of Università Politecnica delle Marche.
%% 
%% Copyright (C) 2011-2012 by Marco Giammarini
%% 
%% This work may be distributed and/or modified under the
%% conditions of the LaTeX Project Public License, either
%% version 1.3 of this license or (at your option) any later
%% version. The latest version of this license is in
%% 
%%   http://www.latex-project.org/lppl.txt
%% 
%% and version 1.3 or later is part of all distributions of LaTeX
%% version 2005/12/01 or later.
%% 
%% This work has the LPPL maintenance status `author-maintained'.
%% 
%% The Current Maintainer of this work is:
%%   Marco Giammarini <m.giammarini@warcomeb.it>
%% _______________________________________________________________
%% 
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{univpmphdthesis}%
  [2011/20/10 v.1.0 Template for PhD thesis of Universita' Politecnica delle Marche]
\errorcontextlines=9
%% Carico ifthen e dichiaro i booleani per le opzioni
\RequirePackage{ifthen,etoolbox}

\newif\ifunivpm@aquattroprint \univpm@aquattroprintfalse
\newif\ifunivpm@english       \univpm@englishfalse
\newif\ifunivpm@italian       \univpm@italianfalse
\newif\ifunivpm@lof           \univpm@loffalse
\newif\ifunivpm@lot           \univpm@lotfalse

%% Dichiaro le opzioni
\DeclareOption{a4print}{\univpm@aquattroprinttrue}
\DeclareOption{english}{\univpm@englishtrue}
\DeclareOption{italian}{\univpm@italiantrue}
\DeclareOption{lof}{\univpm@loftrue}
\DeclareOption{lot}{\univpm@lottrue}

\DeclareOption*{
  \ClassError{univpmphdthesis}%
    {L'opzione \CurrentOption\space non e' valida}%
    {Premi X per terminare e correggi}%
}

%% Carico la classe
\ProcessOptions\relax

\LoadClass[10pt,%
           twoside=true,%
           open=right,%
           cleardoublepage=empty,%
           chapterprefix=true]{scrbook}

%% Gestisco la selezione della lingua
\ifthenelse{\boolean{univpm@english}\AND%
            \boolean{univpm@italian}}%
{%
  \ClassError{univpmphdthesis}%
    {Non puoi definire due lingue!}%
    {Premi X per terminare e correggi l'errore!}%
}%
{\relax}

\ifthenelse{\(\NOT\boolean{univpm@english}\)\AND%
            \(\NOT\boolean{univpm@italian}\)}%
{%
  \ClassError{univpmphdthesis}%
    {Devi definire una lingua!}%
    {Premi X per terminare e correggi l'errore!}%
}%
{\relax}

\typeout{**************************************************}
\ifunivpm@english\typeout{* Hai selezionato: INGLESE}\fi
\ifunivpm@italian\typeout{* Hai selezionato: ITALIANO}\fi
\typeout{**************************************************}

%% \RequirePackage[utf8]{inputenc}
%% \RequirePackage[T1]{fontenc}

\RequirePackage[english,italian]{babel}

\ifunivpm@english
  \AtBeginDocument{\selectlanguage{english}}
\fi

\ifunivpm@italian
  \AtBeginDocument{\selectlanguage{italian}}
  \RequirePackage{indentfirst}
\fi

\addto\captionsitalian{\def\acknowledgename{Ringraziamenti}}
\addto\captionsenglish{\def\acknowledgename{Acknowledgments}}

\addto\captionsitalian{\def\dedicationname{Dedica}}
\addto\captionsenglish{\def\dedicationname{Dedication}}

%% Richiedo il pacchetto per le immagini
\RequirePackage{graphicx}

%% Richiedo il pacchetto per l'interlinea
\RequirePackage{setspace}
\setstretch{1.15}

%% Imposto le dimensioni della pagina
\RequirePackage[twoside]{geometry}
\geometry{%
  papersize={160mm,240mm},
  text={120mm,200mm},
  marginratio=1:1,
  bindingoffset=4mm
}
\addtolength{\footskip}{-0.5cm}

%% Carico hyperref a seconda del driver usato
\RequirePackage{ifpdf}
\ifpdf
  \RequirePackage[pdfpagelabels]{hyperref}
\else
  \RequirePackage{hyperref}
\fi

\ifunivpm@aquattroprint%
  \RequirePackage[a4,
                  center,
                  cam]{crop}
\fi

%% Città dell'autore
\newcommand{\phdlocation}[1]{\def\univpm@phdlocation{#1}}
%% Data di stampa (in qualsiasi formato)
\newcommand{\phdtime}[1]{\def\univpm@phdtime{#1}}
%% Definizione dei comandi per le label in inglese
\newcommand{\authorlabelEN}[1]{\def\univpm@authorlabelEN{#1}}
\newcommand{\advisorlabelEN}[1]{\def\univpm@advisorlabelEN{#1}}
\newcommand{\coadvisorlabelEN}[1]{\def\univpm@coadvisorlabelEN{#1}}
\newcommand{\curriculumadvisorlabelEN}[1]%
  {\def\univpm@curriculumadvisorlabelEN{#1}}
\newcommand{\directorlabelEN}[1]{\def\univpm@directorlabelEN{#1}}
\newcommand{\cyclelabelEN}[1]{\def\univpm@cyclelabelEN{#1}}
%% Label inglesi
\authorlabelEN{Ph.D. Dissertation of:}
\advisorlabelEN{Advisor:}
\coadvisorlabelEN{Coadvisor:}
\curriculumadvisorlabelEN{Curriculum Supervisor:}
\directorlabelEN{Ph.D. School Supervisor:}
\cyclelabelEN{edition - new series}

%% Definizione dei comandi per le label in italiano
\newcommand{\authorlabelIT}[1]{\def\univpm@authorlabelIT{#1}}
\newcommand{\advisorlabelIT}[1]{\def\univpm@advisorlabelIT{#1}}
\newcommand{\coadvisorlabelIT}[1]{\def\univpm@coadvisorlabelIT{#1}}
\newcommand{\curriculumadvisorlabelIT}[1]%
  {\def\univpm@curriculumadvisorlabelIT{#1}}
\newcommand{\directorlabelIT}[1]{\def\univpm@directorlabelIT{#1}}
\newcommand{\cyclelabelIT}[1]{\def\univpm@cyclelabelIT{#1}}
%% Label italiane
\authorlabelIT{Tesi di Dottorato di:}
\advisorlabelIT{Tutor:}
\coadvisorlabelIT{Co-Tutor:}
\curriculumadvisorlabelIT{Coordinatore del Curriculum:}
\directorlabelIT{Coordinatore della Scuola di Dottorato:}
\cyclelabelIT{ciclo - nuova serie}

%% Definizione dei comandi per le label che verranno usate
\newcommand{\authorlabel}[1]{\def\univpm@authorlabel{#1}}
\newcommand{\advisorlabel}[1]{\def\univpm@advisorlabel{#1}}
\newcommand{\coadvisorlabel}[1]{\def\univpm@coadvisorlabel{#1}}
\newcommand{\curriculumadvisorlabel}[1]%
  {\def\univpm@curriculumadvisorlabel{#1}}
\newcommand{\directorlabel}[1]{\def\univpm@directorlabel{#1}}
\newcommand{\cyclelabel}[1]{\def\univpm@cyclelabel{#1}}

%% Scelgo la label giusta in base alla lingua selezionata
\ifunivpm@english%
  \authorlabel{\univpm@authorlabelEN}
  \advisorlabel{\univpm@advisorlabelEN}
  \coadvisorlabel{\univpm@coadvisorlabelEN}
  \curriculumadvisorlabel{\univpm@curriculumadvisorlabelEN}
  \directorlabel{\univpm@directorlabelEN}
  \cyclelabel{\univpm@cyclelabelEN}
\fi

\ifunivpm@italian%
  \authorlabel{\univpm@authorlabelIT}
  \advisorlabel{\univpm@advisorlabelIT}
  \coadvisorlabel{\univpm@coadvisorlabelIT}
  \curriculumadvisorlabel{\univpm@curriculumadvisorlabelIT}
  \directorlabel{\univpm@directorlabelIT}
  \cyclelabel{\univpm@cyclelabelIT}
\fi

%% Nome dell'Università
\newcommand{\univpm}[1]{\def\univpm@univpm{#1}}
\univpm{Universit\`{a} Politecnica delle Marche}
%% Indirizzo dell'Università
\newcommand{\facultyaddress}[1]{\def\univpm@facultyaddress{#1}}
\facultyaddress{Via Brecce Bianche -- 60131 Ancona (AN), Italy}
%% Logo dell'Università
\def\univpm@logo{figures/logoUNIVPM}
%%  Nome della scuola di dottorato
\newcommand{\phdschool}[1]{\def\univpm@phdschool{#1}}
%%  Facoltà della scuola di dottorato
\newcommand{\phdfaculty}[1]{\def\univpm@phdfaculty{#1}}
%% Nome del curriculum
\newcommand{\phdcurriculum}[1]{\def\univpm@phdcurriculum{#1}}
%% Titolo della tesi
\newcommand{\phdtitle}[1]{\def\univpm@phdtitle{#1}}
%% Sotto titolo della tesi
\newcommand{\phdsubtitle}[1]{\def\univpm@phdsubtitle{#1}}
%% Autore della tesi
\newcommand{\phdauthor}[1]{\def\univpm@phdauthor{#1}}
%% Tutor del dottorando
\newcommand{\phdadvisor}[1]{\def\univpm@phdadvisor{#1}}
%% Co-Tutor del dottorando
\newcommand{\phdcoadvisor}[1]{\def\univpm@phdcoadvisor{#1}}
%% Coordinatore del curriculum
\newcommand{\phdcurriculumadvisor}[1]%
  {\def\univpm@phdcurriculumadvisor{#1}}

%% Comando per la gestione del ciclo di dottorato
\newcounter{univpm@phdcyclenumber}
\newcommand{\phdcycle}[1]%
{%
  \setcounter{univpm@phdcyclenumber}{#1}
  \def\univpm@phdcycle{%
    \Roman{univpm@phdcyclenumber} \univpm@cyclelabel}
}
%% Comando per la gestione della dedica
\newcommand{\thesisdedication}[1]%
{%
  \def\univpm@dedication{#1}
}
%% Comando per la gestione dell'hardcover
\newcommand{\hscover}%
{%
%% Logo e nomi dell'Università
  \begin{minipage}[t]{\textwidth}
    \begin{center}
      \includegraphics[scale=.3]{\univpm@logo}\par
      \small{
        \textsc{\univpm@univpm}\par
        \textsc{\univpm@phdschool}\par
        \textsc{\univpm@phdcurriculum}\par
      }
      \vspace{1em}
      \hrule
    \end{center}
  \end{minipage}

  \vspace{3em}

%% Titolo
  \begin{minipage}[t]{\textwidth}
    \begin{center}
      \huge\usekomafont{title}{\univpm@phdtitle}\par
      \ifdefined\univpm@phdsubtitle
        \vspace{2em}
        \large\usekomafont{title}{\univpm@phdsubtitle}\par
      \fi
    \end{center}
  \end{minipage}

  \vfill

%% Autore, tutor e cordinatore
  \begin{minipage}[t]{\textwidth}
    \begin{flushright}
      \large{%
        \univpm@authorlabel\par
      \textbf{\univpm@phdauthor}\par
      }
    \end{flushright}
    \vspace{1em}
    \univpm@advisorlabel\par
    \textbf{\univpm@phdadvisor}\par
    \vspace{2em}
    \ifdefined\univpm@phdcoadvisor
      \univpm@coadvisorlabel\par
      \textbf{\univpm@phdcoadvisor}\par
      \vspace{2em}
    \fi
    \univpm@curriculumadvisorlabel\par
  \textbf{\univpm@phdcurriculumadvisor}\par
  \end{minipage}

  \vspace{2em}

%% Numero di ciclo
  \begin{minipage}[t]{\textwidth}
    \begin{center}
      \univpm@phdcycle\par
    \end{center}
  \end{minipage}
}

%% Comando per la gestione del colophon
\newcommand{\colophon}%
{%
  \hfill\vfill
  \noindent\begin{minipage}[t]{\textwidth}
    \hrule
    \vspace{1em}
    \begin{center}
      \small{
        \textsc{\univpm@univpm}\par
        \textsc{\univpm@phdschool}\par
        \textsc{\univpm@phdfaculty}\par
        \univpm@facultyaddress\par
      }
    \end{center}
  \end{minipage}
}
%% Viene ridefinito il comando maketitle
\renewcommand*\maketitle{%
%% Doppia pagina vuota
  \hfill\vfill
  \cleardoublepage\thispagestyle{empty}
%% Hardcover e pagina vuota
  \ifpdf\pdfbookmark{Hardcover}{Hardcover}\fi
  \hscover
  \cleardoublepage\thispagestyle{empty}
%% Softcover e colophon
  \ifpdf\pdfbookmark{Softcover}{Softcover}\fi
  \hscover
  \clearpage\thispagestyle{empty}
  \colophon
%% Dedica, se presente
  \ifdefined\univpm@dedication
    \cleardoublepage\thispagestyle{empty}
    \ifpdf\pdfbookmark{\dedicationname}{\dedicationname}\fi
    \hfill\vfill
    \begin{flushright}
      \large\textit{\univpm@dedication}
    \end{flushright}
    \vfill
  \fi
}
%% Ambiente per gestire i ringraziamenti
\newenvironment{thesisacknowledge}[1][]%
{%
%% In base alla lingua selezionata creo titolo e testo
%%  \selectlanguage{#1}
  \ifblank{#1}{}{\begin{otherlanguage}{#1}}
    \chapter*{\acknowledgename}
    \ifpdf
      \pdfbookmark{\acknowledgename}{\acknowledgename}
    \fi
  \ifblank{#1}{}{\end{otherlanguage}}
}%
{%
%% Stampo data e firma dell'autore
    \ifdefined\univpm@phdlocation
      \ifdefined\univpm@phdtime
        \bigskip\par
        \noindent\textit{\univpm@phdlocation, \univpm@phdtime}\par
        \hfill\univpm@phdauthor\par
      \else
        \ClassError{univpmphdthesis}%
          {Devi definire la data di stampa!}%
          {Premi X per terminare e correggi l'errore!}%
      \fi
    \else
      \ClassError{univpmphdthesis}%
        {Devi definire il luogo di stampa!}%
        {Premi X per terminare e correggi l'errore!}%
    \fi
%% Ripristino la lingua
%%  \ifunivpm@english
%%    \selectlanguage{english}
%%  \fi

%%  \ifunivpm@italian
%%    \selectlanguage{italian}
%%  \fi
}
%% Ambiente per gestire l'abstract
\newenvironment{thesisabstract}[1][]%
{%
%% In base alla lingua selezionata creo titolo e testo
%%  \selectlanguage{#1}
  \ifblank{#1}{}{\begin{otherlanguage}{#1}}
    \chapter*{\abstractname}
    \ifpdf\pdfbookmark{\abstractname}{\abstractname}\fi
  \ifblank{#1}{}{\end{otherlanguage}}
}
{%
%% Ripristino la lingua
%%  \ifunivpm@english
%%    \selectlanguage{english}
%%  \fi

%%  \ifunivpm@italian
%%    \selectlanguage{italian}
%%  \fi
\relax}
%% Ambiente per gestire i ringraziamenti
\newcommand{\thesistoc}%
{%
  \cleardoublepage
  \ifpdf\pdfbookmark{\contentsname}{\contentsname}\fi
  \tableofcontents

  \ifunivpm@lof%
    \cleardoublepage
    \ifpdf\pdfbookmark{\listfigurename}{\listfigurename}\fi
    \listoffigures
  \fi

  \ifunivpm@lot%
    \cleardoublepage
    \ifpdf\pdfbookmark{\listtablename}{\listtablename}\fi
    \listoftables
  \fi
}

\endinput
%%
%% End of file `univpmphdthesis.cls'.
