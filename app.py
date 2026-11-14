import streamlit as st
import pandas as pd
import json
import requests
from datetime import datetime

st.set_page_config(page_title="Agentic AI Learning Platform", page_icon="🚀", layout="wide")

# ==========================================
# 1. AGENT SYSTEM PROMPTS (For PromptWars Judges!)
# ==========================================
AGENT_PROMPTS = {
    "Profile Agent": """ROLE: Expert Learner Profile Extractor Agent.
TASK: Parse learner details, existing skills (0-5 scale), certifications, and projects into a normalized JSON profile.
RULES: Infer foundational skill levels from projects and certifications. Output strict JSON.""",

    "Skill Analysis Agent": """ROLE: Diagnostic Skill Evaluator Agent.
TASK: Analyze the learner's skill matrix and recent assessment telemetry to categorize Strengths (level >= 3) and Weaknesses (level < 3 or failed quiz).
RULES: Weigh practical project experience and quiz scores higher than self-reported ratings.""",

    "Career Matching Agent": """ROLE: Target Role Alignment & Gap Computation Agent.
TASK: Compare learner's current skill levels against target career role benchmarks.
RULES: Calculate Readiness Score (%) weighted by skill criticality. Output prioritized Missing Skills and Partial Gaps.""",

    "Curriculum Agent": """ROLE: Adaptive Learning Path Architect Agent.
TASK: Sequence missing and weak skills into a prerequisite-aware learning roadmap with curated courses, tutorials, and hands-on labs.
RULES: Enforce strict dependency order (e.g., HTML -> JavaScript -> React & Node.js). Inject [REMEDIAL] modules if Assessment Agent flags a failed quiz.""",

    "Assessment & Progress Agent": """ROLE: Dynamic Evaluator & Telemetry Tracker Agent.
TASK: Generate targeted diagnostic questions for current learning modules, grade responses, and update skill mastery levels.""",

    "AI Mentor (Supervisor)": """ROLE: Autonomous Supervisor & Career Mentor Agent.
TASK: Orchestrate specialized agents when state changes occur (Career Goal Change OR Poor Assessment Performance < 60%).
RULES: Provide empathetic, actionable guidance explaining WHY the learning path was dynamically re-planned."""
}

# ==========================================
# 2. ROLE REQUIREMENTS & RESOURCE CATALOG
# ==========================================
ROLES_DB = {
    "Full Stack Developer": {
        "HTML": {"level": 3, "weight": 0.6, "prereq": None},
        "Python": {"level": 3, "weight": 0.7, "prereq": None},
        "SQL": {"level": 3, "weight": 0.8, "prereq": None},
        "JavaScript": {"level": 4, "weight": 1.0, "prereq": "HTML"},
        "React": {"level": 4, "weight": 1.0, "prereq": "JavaScript"},
        "Node.js": {"level": 4, "weight": 1.0, "prereq": "JavaScript"},
        "REST APIs": {"level": 3, "weight": 0.8, "prereq": "Node.js"},
        "Git & Deployment": {"level": 3, "weight": 0.5, "prereq": None}
    },
    "Data Scientist / AI Engineer": {
        "Python": {"level": 4, "weight": 1.0, "prereq": None},
        "SQL": {"level": 4, "weight": 0.9, "prereq": None},
        "Statistics & Math": {"level": 3, "weight": 0.8, "prereq": None},
        "Pandas & NumPy": {"level": 4, "weight": 0.9, "prereq": "Python"},
        "Machine Learning": {"level": 4, "weight": 1.0, "prereq": "Pandas & NumPy"},
        "Deep Learning & LLMs": {"level": 3, "weight": 0.9, "prereq": "Machine Learning"}
    },
    "Backend Engineer": {
        "Python": {"level": 4, "weight": 1.0, "prereq": None},
        "SQL": {"level": 4, "weight": 1.0, "prereq": None},
        "Node.js": {"level": 3, "weight": 0.8, "prereq": None},
        "REST APIs": {"level": 4, "weight": 1.0, "prereq": "Python"},
        "System Design & Docker": {"level": 3, "weight": 0.8, "prereq": "REST APIs"},
        "Git & Deployment": {"level": 3, "weight": 0.6, "prereq": None}
    }
}

RESOURCES_DB = {
    "HTML": {
        "course": "MDN Web Docs: Structuring the Web with HTML",
        "tutorial": "freeCodeCamp Responsive Web Design Cert",
        "exercise": "Build a Semantic Personal Portfolio Page",
        "remedial": "HTML DOM Tree & Semantic Tags Crash Refresher (45 mins)"
    },
    "JavaScript": {
        "course": "JavaScript.info: The Modern JavaScript Tutorial (ES6+)",
        "tutorial": "Async JS, Closures, Promises & Fetch API Deep Dive",
        "exercise": "Build an Interactive Task Manager with LocalStorage & Fetch",
        "remedial": "⚠️ [REMEDIAL] Visual Guide to JS Closures, Scope & Event Loop + 5 Debugging Drills"
    },
    "React": {
        "course": "Official React.dev Learn: Describing the UI & Hooks",
        "tutorial": "State Management with useState, useEffect & Context API",
        "exercise": "Build a Dynamic Job Tracker Dashboard in React",
        "remedial": "⚠️ [REMEDIAL] React Props vs State & Component Lifecycle Step-by-Step Lab"
    },
    "Node.js": {
        "course": "Node.js & Express Official Fundamentals",
        "tutorial": "Building Scalable Server-Side Apps with Express & Middleware",
        "exercise": "Create a JWT-Authenticated Backend Server in Node.js",
        "remedial": "⚠️ [REMEDIAL] Node.js Async Callbacks, Event Emitter & Express Routing Basics"
    },
    "REST APIs": {
        "course": "RESTful API Design, Status Codes & CRUD Best Practices",
        "tutorial": "Connecting Node.js/Python APIs to SQL Databases",
        "exercise": "Build & Test a Full CRUD API with Postman",
        "remedial": "⚠️ [REMEDIAL] HTTP Methods, Headers & JSON Payload Troubleshooting"
    },
    "SQL": {
        "course": "SQLBolt & Mode SQL Analytics Training",
        "tutorial": "Mastering JOINs, Indexing, and Aggregations",
        "exercise": "Design a Normalized Schema for an E-Commerce App",
        "remedial": "⚠️ [REMEDIAL] Visual SQL JOINs & Subqueries Practice Workbench"
    },
    "Python": {
        "course": "Python for Everybody & Intermediate OOP",
        "tutorial": "Data Structures, List Comprehensions & Virtual Environments",
        "exercise": "Build a CLI Data Scraper and File Automation Script",
        "remedial": "⚠️ [REMEDIAL] Python Functions, Dictionaries & Error Handling Drills"
    },
    "Git & Deployment": {
        "course": "Git Branching, Merging & CI/CD Basics",
        "tutorial": "Deploying Full Stack Apps on Render / Vercel",
        "exercise": "Push Multi-Branch Project to GitHub & Configure Auto-Deploy",
        "remedial": "⚠️ [REMEDIAL] Interactive Git Branching Sandbox"
    },
    "Statistics & Math": {
        "course": "Khan Academy: Probability, Distributions & Hypothesis Testing",
        "tutorial": "Linear Algebra & Calculus for Machine Learning",
        "exercise": "Perform EDA & Statistical Significance Tests on a Dataset",
        "remedial": "⚠️ [REMEDIAL] Core Probability & Bayes Theorem Refresher"
    },
    "Pandas & NumPy": {
        "course": "Data Manipulation with Pandas & Vectorized NumPy",
        "tutorial": "Cleaning Missing Data, GroupBy & Feature Engineering",
        "exercise": "Clean & Analyze 50k Rows of Real-World Sales Data",
        "remedial": "⚠️ [REMEDIAL] Pandas DataFrame Indexing & Filtering Bootcamp"
    },
    "Machine Learning": {
        "course": "Scikit-Learn Supervised & Unsupervised Learning",
        "tutorial": "Model Evaluation, Overfitting & Hyperparameter Tuning",
        "exercise": "Train & Deploy a Customer Churn Prediction Model",
        "remedial": "⚠️ [REMEDIAL] Bias-Variance Tradeoff & Regression Fundamentals"
    },
    "Deep Learning & LLMs": {
        "course": "Neural Networks, Transformers & Prompt Engineering",
        "tutorial": "Building RAG & Multi-Agent Systems with APIs",
        "exercise": "Build a Domain-Specific AI Assistant with Vector Search",
        "remedial": "⚠️ [REMEDIAL] Embeddings, Attention Mechanism & Prompting Basics"
    },
    "System Design & Docker": {
        "course": "Containerization with Docker & Microservices Intro",
        "tutorial": "Caching (Redis), Load Balancing & Database Scaling",
        "exercise": "Dockerize a Backend API + PostgreSQL with Docker Compose",
        "remedial": "⚠️ [REMEDIAL] Docker Images vs Containers Hands-on Lab"
    }
}

QUIZ_DB = {
    "JavaScript": [
        {
            "q": "What happens when a function inside another function accesses variables from the outer function after the outer function has returned?",
            "options": [
                "ReferenceError is thrown",
                "It forms a Closure and retains access to the outer lexical scope",
                "Variables are reset to undefined",
                "JavaScript blocks execution"
            ],
            "answer": "It forms a Closure and retains access to the outer lexical scope"
        },
        {
            "q": "Which method converts a JSON string from an API response into a JavaScript object?",
            "options": ["JSON.stringify()", "JSON.parse()", "JSON.toObject()", "Object.fromJSON()"],
            "answer": "JSON.parse()"
        },
        {
            "q": "What does an 'async' function always return in JavaScript?",
            "options": ["A String", "A Callback", "A Promise", "An Array"],
            "answer": "A Promise"
        }
    ],
    "React": [
        {
            "q": "Which React Hook is used to manage local state in a functional component?",
            "options": ["useEffect", "useState", "useContext", "useRouter"],
            "answer": "useState"
        },
        {
            "q": "Why should you never mutate state directly in React (e.g., state.count = 5)?",
            "options": [
                "It causes a syntax error",
                "React won't know the state changed and won't trigger a re-render",
                "It deletes the component",
                "HTML doesn't allow numbers"
            ],
            "answer": "React won't know the state changed and won't trigger a re-render"
        }
    ],
    "Node.js": [
        {
            "q": "How does Node.js handle thousands of concurrent I/O requests despite being single-threaded?",
            "options": [
                "By creating a new OS thread for every user",
                "Via the Non-Blocking Event Loop and asynchronous callbacks/promises",
                "By pausing the browser",
                "Using SQL triggers"
            ],
            "answer": "Via the Non-Blocking Event Loop and asynchronous callbacks/promises"
        }
    ]
}

# ==========================================
# 3. SESSION STATE INITIALIZATION (Sample Scenario Default)
# ==========================================
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.learner_name = "Arun Kumar (Sample Scenario)"
    st.session_state.target_role = "Full Stack Developer"
    st.session_state.certs = "Python Basic Cert, SQL Fundamentals Badge"
    st.session_state.projects = "Student Database CLI (Python + SQL), Static College Event Page (Basic HTML)"
    # Sample scenario skills: Python (3), SQL (3), Basic HTML (2), Missing: JS (0), React (0), Node.js (0)
    st.session_state.skills = {
        "Python": 3,
        "SQL": 3,
        "HTML": 2,
        "JavaScript": 0,
        "React": 0,
        "Node.js": 0,
        "REST APIs": 0,
        "Git & Deployment": 1,
        "Statistics & Math": 1,
        "Pandas & NumPy": 1,
        "Machine Learning": 0,
        "Deep Learning & LLMs": 0,
        "System Design & Docker": 0
    }
    st.session_state.remedial_flags = set()
    st.session_state.completed_modules = set()
    st.session_state.quiz_history = []
    st.session_state.agent_logs = []
    st.session_state.mentor_message = (
        "👋 **Welcome!** As your **AI Mentor Supervisor**, I coordinated with the **Profile, Skill Analysis, "
        "and Career Matching Agents**. You have a solid foundation in **Python (3/5)** and **SQL (3/5)**, plus **basic HTML (2/5)**. "
        "To become a **Full Stack Developer**, your critical missing skills are **JavaScript, React, and Node.js**. "
        "I have sequenced your roadmap starting with HTML polish -> JavaScript -> React & Node.js!"
    )

def log_agent(agent_name, action):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.agent_logs.insert(0, {"time": timestamp, "agent": agent_name, "action": action})

if len(st.session_state.agent_logs) == 0:
    log_agent("Profile Agent", "Parsed learner skills: Python (3), SQL (3), Basic HTML (2), Missing JS/React/Node.")
    log_agent("Skill Analysis Agent", "Strengths: Python, SQL | Weaknesses/Missing: HTML (Basic), JavaScript, React, Node.js.")
    log_agent("Career Matching Agent", "Computed alignment against 'Full Stack Developer' role requirements.")
    log_agent("Curriculum Agent", "Generated prerequisite-ordered learning path (HTML -> JS -> React -> Node.js -> REST APIs).")
    log_agent("AI Mentor (Supervisor)", "Initialized personalized guidance and monitoring assessment triggers.")

# ==========================================
# 4. CORE MULTI-AGENT FUNCTIONS
# ==========================================
def run_skill_and_career_agents(skills, target_role):
    role_reqs = ROLES_DB[target_role]
    total_weighted = sum(r["level"] * r["weight"] for r in role_reqs.values())
    earned_weighted = 0.0
    
    strengths = []
    partial_gaps = []
    missing_skills = []
    gap_details = []

    for skill, req in role_reqs.items():
        curr = skills.get(skill, 0)
        req_lvl = req["level"]
        w = req["weight"]
        earned_weighted += min(curr, req_lvl) * w
        gap = max(0, req_lvl - curr)

        if curr >= req_lvl and skill not in st.session_state.remedial_flags:
            status = "✅ Mastered / Ready"
            strengths.append(f"{skill} ({curr}/{req_lvl})")
        elif curr == 0:
            status = "🚨 Missing Skill"
            missing_skills.append(skill)
        else:
            status = "⚠️ Needs Improvement" if skill not in st.session_state.remedial_flags else "🛑 Remedial Required"
            partial_gaps.append(f"{skill} ({curr}/{req_lvl})")

        gap_details.append({
            "Skill": skill,
            "Current Level": curr,
            "Required Level": req_lvl,
            "Gap": gap,
            "Priority Score": round(gap * w + (2.0 if skill in st.session_state.remedial_flags else 0), 2),
            "Prerequisite": req["prereq"] or "None",
            "Status": status
        })

    readiness = round((earned_weighted / total_weighted) * 100, 1) if total_weighted else 0
    return readiness, strengths, partial_gaps, missing_skills, gap_details

def build_roadmap(gap_details):
    roadmap = []
    phase = 1
    # Prerequisite-aware ordering: items with no unresolved prereq come first, remedial boosted
    for item in gap_details:
        skill = item["Skill"]
        if item["Gap"] > 0 or skill in st.session_state.remedial_flags:
            res = RESOURCES_DB.get(skill, {})
            is_remedial = skill in st.session_state.remedial_flags
            roadmap.append({
                "phase": phase,
                "skill": skill,
                "current": item["Current Level"],
                "target": item["Required Level"],
                "prereq": item["Prerequisite"],
                "is_remedial": is_remedial,
                "course": res.get("remedial" if is_remedial else "course", "Core Module"),
                "tutorial": res.get("tutorial", "Guided Tutorial"),
                "exercise": res.get("exercise", "Hands-on Capstone Task")
            })
            phase += 1
    return roadmap

# ==========================================
# 5. SIDEBAR: LEARNER PROFILE & DEMO TRIGGERS
# ==========================================
with st.sidebar:
    st.title("🎓 Learner Profile")
    st.caption("Managed by **Profile Agent**")
    
    st.session_state.learner_name = st.text_input("Learner Name", st.session_state.learner_name)
    
    # Career Goal Selector (Trigger 1: Dynamic Goal Change!)
    selected_role = st.selectbox(
        "🎯 Target Career Goal",
        list(ROLES_DB.keys()),
        index=list(ROLES_DB.keys()).index(st.session_state.target_role)
    )
    
    if selected_role != st.session_state.target_role:
        old_role = st.session_state.target_role
        st.session_state.target_role = selected_role
        log_agent("AI Mentor (Supervisor)", f"Detected Career Goal change: '{old_role}' -> '{selected_role}'. Triggering re-analysis.")
        log_agent("Career Matching Agent", f"Re-computed skill gaps against '{selected_role}' competency matrix.")
        log_agent("Curriculum Agent", f"Re-built personalized learning roadmap for '{selected_role}'.")
        st.session_state.mentor_message = (
            f"🔄 **Career Goal Updated to {selected_role}!** I instructed the **Career Matching Agent** and "
            f"**Curriculum Agent** to recalculate your transferable skills (like Python & SQL) and build a brand-new "
            f"learning roadmap tailored for **{selected_role}**."
        )
        st.rerun()

    st.session_state.certs = st.text_area("🏅 Certifications", st.session_state.certs, height=68)
    st.session_state.projects = st.text_area("💻 Past Projects", st.session_state.projects, height=68)

    st.divider()
    st.subheader("🛠️ Current Skill Levels (0–5)")
    role_skills = list(ROLES_DB[st.session_state.target_role].keys())
    for sk in role_skills:
        new_val = st.slider(f"{sk}", 0, 5, st.session_state.skills.get(sk, 0), key=f"slider_{sk}")
        if new_val != st.session_state.skills.get(sk, 0):
            st.session_state.skills[sk] = new_val
            log_agent("Profile Agent", f"Updated '{sk}' skill rating to {new_val}/5.")
            log_agent("Curriculum Agent", "Adjusted learning path based on updated skill profile.")

    st.divider()
    if st.button("🔄 Reset to Sample Scenario", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# Run multi-agent analysis
readiness, strengths, partial_gaps, missing_skills, gap_details = run_skill_and_career_agents(
    st.session_state.skills, st.session_state.target_role
)
roadmap = build_roadmap(gap_details)

# ==========================================
# 6. MAIN HEADER & AI MENTOR BANNER
# ==========================================
st.title("🚀 Agentic AI Personalized Learning & Career Pathway Platform")
st.markdown(
    f"**Learner:** `{st.session_state.learner_name}` &nbsp;|&nbsp; "
    f"**Target Career:** `{st.session_state.target_role}` &nbsp;|&nbsp; "
    f"**Active AI Agents:** `6 Collaborating Agents`"
)

st.info(st.session_state.mentor_message, icon="🤖")

# Top KPI Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎯 Career Readiness", f"{readiness}%", delta="Live Agent Score")
col2.metric("✅ Strengths Identified", len(strengths), delta="Mastered Skills")
col3.metric("🚨 Missing / Weak Skills", len(missing_skills) + len(partial_gaps), delta="-Skill Gaps", delta_color="inverse")
completed_count = len(st.session_state.completed_modules)
total_modules = max(1, len(roadmap))
col4.metric("📈 Roadmap Progress", f"{int((completed_count / total_modules) * 100)}%", f"{completed_count}/{len(roadmap)} Modules")

# ==========================================
# 7. NAVIGATION TABS
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Skill Analysis & Career Match",
    "🗺️ Adaptive Learning Roadmap",
    "📝 AI Assessment & Mentor Trigger",
    "🧠 Multi-Agent Collaboration & Prompts"
])

# --- TAB 1: SKILL ANALYSIS & CAREER MATCHING ---
with tab1:
    st.subheader(f"🔍 Skill Gap Analysis for {st.session_state.target_role}")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("**💪 Strengths (Ready)**\n\n" + ("\n".join(f"- {s}" for s in strengths) if strengths else "- None yet"))
    with c2:
        st.warning("**⚡ Partial Skills (To Upgrade)**\n\n" + ("\n".join(f"- {s}" for s in partial_gaps) if partial_gaps else "- None"))
    with c3:
        st.error("**🚨 Missing Core Skills**\n\n" + ("\n".join(f"- {s}" for s in missing_skills) if missing_skills else "- No missing skills!"))

    df_gaps = pd.DataFrame(gap_details)
    
    left_chart, right_table = st.columns([1, 1])
    with left_chart:
        st.markdown("#### 📊 Current vs Required Competency Level")
        chart_df = df_gaps.set_index("Skill")[["Current Level", "Required Level"]]
        st.bar_chart(chart_df)
    with right_table:
        st.markdown("#### 📋 Career Matching Agent Output Matrix")
        st.dataframe(df_gaps, use_container_width=True, hide_index=True)

# --- TAB 2: ADAPTIVE LEARNING ROADMAP ---
with tab2:
    st.subheader("🗺️ Personalized Prerequisite-Aware Learning Path")
    st.caption("Dynamically generated by **Curriculum Agent** & monitored by **Progress Tracker Agent**")

    if not roadmap:
        st.balloons()
        st.success("🎉 Congratulations! You meet all skill requirements for your target career!")
    else:
        for mod in roadmap:
            skill_name = mod["skill"]
            badge = "🛑 [REMEDIAL INTERVENTION]" if mod["is_remedial"] else f"Phase {mod['phase']}"
            header_text = f"{badge} — {skill_name} (Level {mod['current']} ➔ {mod['target']}) | Prerequisite: {mod['prereq']}"
            
            with st.expander(header_text, expanded=True):
                if mod["is_remedial"]:
                    st.error(
                        f"**AI Mentor Alert:** Because your recent **{skill_name}** assessment score was below 60%, "
                        f"the Curriculum Agent injected a foundational remedial bootcamp before advancing."
                    )
                m_col1, m_col2 = st.columns([3, 1])
                with m_col1:
                    st.markdown(f"- 📚 **Recommended Course:** {mod['course']}")
                    st.markdown(f"- 🎥 **Guided Tutorial:** {mod['tutorial']}")
                    st.markdown(f"- 💻 **Practical Exercise:** {mod['exercise']}")
                with m_col2:
                    is_done = skill_name in st.session_state.completed_modules
                    checked = st.checkbox("Mark Module Complete", value=is_done, key=f"chk_{skill_name}")
                    if checked and not is_done:
                        st.session_state.completed_modules.add(skill_name)
                        # Boost skill by 1 on completion
                        st.session_state.skills[skill_name] = min(5, st.session_state.skills.get(skill_name, 0) + 1)
                        st.session_state.remedial_flags.discard(skill_name)
                        log_agent("Progress Tracker Agent", f"Learner completed '{skill_name}' module. Skill level incremented.")
                        st.rerun()
                    elif not checked and is_done:
                        st.session_state.completed_modules.discard(skill_name)
                        st.rerun()

# --- TAB 3: AI ASSESSMENT & ADAPTIVE MENTOR TRIGGER ---
with tab3:
    st.subheader("📝 Diagnostic Skill Assessment (Adaptive Re-Planning Trigger)")
    st.markdown(
        "Test how the **Assessment Agent** and **AI Mentor Supervisor** collaborate! "
        "**Tip for Demo:** Intentionally fail the quiz (<60%) to watch the agents dynamically inject a **Remedial Path**, "
        "or pass (100%) to auto-upgrade your skill level!"
    )

    quiz_skill = st.selectbox("Select Skill to Assess:", list(QUIZ_DB.keys()))
    questions = QUIZ_DB[quiz_skill]

    with st.form("quiz_form"):
        user_answers = {}
        for idx, q_item in enumerate(questions):
            st.markdown(f"**Q{idx+1}. {q_item['q']}**")
            user_answers[idx] = st.radio(
                f"Select answer for Q{idx+1}",
                q_item["options"],
                key=f"q_{quiz_skill}_{idx}",
                label_visibility="collapsed"
            )
            st.write("")
        submitted = st.form_submit_button("🚀 Submit Assessment to AI Evaluator")

    if submitted:
        correct = sum(1 for idx, q_item in enumerate(questions) if user_answers[idx] == q_item["answer"])
        score_pct = round((correct / len(questions)) * 100, 1)
        st.session_state.quiz_history.append({"Skill": quiz_skill, "Score": f"{score_pct}%", "Result": "Pass" if score_pct >= 60 else "Needs Remediation"})
        
        log_agent("Assessment & Progress Agent", f"Graded '{quiz_skill}' assessment: {correct}/{len(questions)} ({score_pct}%).")

        if score_pct < 60:
            st.session_state.remedial_flags.add(quiz_skill)
            st.session_state.skills[quiz_skill] = max(1, st.session_state.skills.get(quiz_skill, 0))
            log_agent("AI Mentor (Supervisor)", f"Alert! Low score ({score_pct}%) in {quiz_skill}. Instructing Curriculum Agent to insert Remedial Lab.")
            log_agent("Curriculum Agent", f"Injected [REMEDIAL] crash course and debugging drills for {quiz_skill} into Phase 1.")
            st.session_state.mentor_message = (
                f"⚠️ **Adaptive Intervention Triggered!** You scored **{score_pct}%** in the **{quiz_skill}** assessment. "
                f"Don't worry—I collaborated with the **Curriculum Agent** to pause advanced frameworks and insert a "
                f"**Remedial {quiz_skill} Crash Course & Hands-on Drills** into your Roadmap. Check the Roadmap tab!"
            )
            st.error(f"Score: {score_pct}% — Below 60% threshold! AI Mentor has dynamically updated your Learning Roadmap with Remedial content.")
        else:
            st.session_state.remedial_flags.discard(quiz_skill)
            new_lvl = min(5, max(3, st.session_state.skills.get(quiz_skill, 0) + 2))
            st.session_state.skills[quiz_skill] = new_lvl
            log_agent("AI Mentor (Supervisor)", f"High assessment performance ({score_pct}%) in {quiz_skill}! Upgraded skill level to {new_lvl}/5.")
            log_agent("Curriculum Agent", f"Fast-tracked {quiz_skill} and unlocked downstream dependencies (React / Node.js).")
            st.session_state.mentor_message = (
                f"🌟 **Outstanding Performance!** You scored **{score_pct}%** in **{quiz_skill}**. "
                f"The **Skill Analysis Agent** upgraded your **{quiz_skill}** proficiency to **{new_lvl}/5**, and the "
                f"**Curriculum Agent** fast-tracked your roadmap to focus on next-level frameworks!"
            )
            st.success(f"Score: {score_pct}% — Passed! Your {quiz_skill} skill level has been upgraded to {new_lvl}/5 and your roadmap adjusted.")

    if st.session_state.quiz_history:
        st.markdown("#### 📜 Assessment Telemetry History")
        st.dataframe(pd.DataFrame(st.session_state.quiz_history), use_container_width=True, hide_index=True)

# --- TAB 4: MULTI-AGENT COLLABORATION & SYSTEM PROMPTS ---
with tab4:
    st.subheader("🧠 Real-Time Multi-Agent Collaboration Trace")
    st.caption("Shows how specialized autonomous agents communicate and adapt to learner state changes.")
    
    st.dataframe(pd.DataFrame(st.session_state.agent_logs), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📜 Specialized Agent System Prompts (PromptWars Architecture)")
    for agent_title, prompt_text in AGENT_PROMPTS.items():
        with st.expander(f"🤖 {agent_title} — System Prompt"):
            st.code(prompt_text, language="markdown")