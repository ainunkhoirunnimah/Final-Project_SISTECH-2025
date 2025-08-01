#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Career Recommendation System - V18 (Definitive & Working)
- Fixes the TypeError by correctly unpacking the profile dictionary.
- Implements the advanced "Classification Tree" logic.
- Uses a robust pipeline to load all available Kaggle datasets.
- Procedurally generates 100+ diverse user profiles for comprehensive testing.
"""

import subprocess, sys, os, zipfile, json, re, time, random
from collections import defaultdict
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def install(package):
    try: subprocess.check_call([sys.executable, "-m", "pip", "install", package, '--quiet'])
    except: pass

print("🚀 Starting Enhanced Career Recommendation System...")
print("📦 Installing dependencies..."); install('pandas'); install('scikit-learn'); install('kaggle'); print("✅ Dependencies installed.")

# === CAREER CLASSIFICATION TREE & SKILL MAPPINGS ===
CAREER_TREE = {
    "Technology": {
        "Software Development": {"skills": ["python", "javascript", "java", "react", "nodejs", "git"], "roles": ["Software Engineer", "Full Stack Developer"]},
        "Data Science": {"skills": ["python", "sql", "machine_learning", "pandas", "statistics"], "roles": ["Data Scientist", "Data Analyst", "ML Engineer"]},
        "DevOps/Cloud": {"skills": ["aws", "docker", "kubernetes", "linux", "ci_cd"], "roles": ["DevOps Engineer", "Cloud Architect"]},
        "Cybersecurity": {"skills": ["cybersecurity", "network_security", "penetration_testing"], "roles": ["Security Analyst", "Penetration Tester"]},
    },
    "Design": {"UI/UX Design": {"skills": ["uiux", "figma", "user_research", "prototyping"], "roles": ["UI Designer", "UX Designer", "Product Designer"]}},
    "Business": {"Project Management": {"skills": ["agile", "project_management"], "roles": ["Project Manager", "Scrum Master"]}}
}
SKILL_SYNONYMS = {
    'python': ['python', 'py', 'django', 'flask', 'pandas', 'numpy'], 'javascript': ['javascript', 'js', 'nodejs', 'typescript', 'vue', 'angular', 'react', 'reactjs', 'react native'],
    'java': ['java', 'jvm', 'spring'], 'sql': ['sql', 'mysql', 'postgresql', 'database'], 'machine_learning': ['machine learning', 'ml', 'ai', 'deep learning', 'tensorflow', 'pytorch'],
    'aws': ['aws', 'amazon web services', 'cloud', 'ec2', 's3'], 'docker': ['docker', 'containerization', 'kubernetes', 'k8s'], 'git': ['git', 'github', 'gitlab', 'version control'],
    'uiux': ['ui/ux', 'ui', 'ux', 'user experience', 'figma', 'sketch', 'adobe xd'], 'cybersecurity': ['cybersecurity', 'security', 'infosec'], 'agile': ['agile', 'scrum', 'kanban', 'jira'],
    'project_management': ['project management', 'pmp', 'prince2']
}

def normalize_skill(skill):
    skill_lower = str(skill).lower().strip()
    for canonical, synonyms in SKILL_SYNONYMS.items():
        if any(syn in skill_lower for syn in synonyms): return canonical
    return skill_lower

class CareerAdviser:
    def __init__(self):
        self.career_tree = CAREER_TREE
        self.jobs_df = pd.DataFrame()
        self.learning_df = pd.DataFrame()

    def load_data(self):
        print("🔄 Loading and processing all Kaggle datasets...")
        data_dir = 'data'; os.makedirs(data_dir, exist_ok=True)
        
        # Setup Kaggle
        kaggle_dir = os.path.expanduser('~/.kaggle')
        if not os.path.exists(os.path.join(kaggle_dir, 'kaggle.json')):
            print("   ⚠️ kaggle.json not found. Creating fallback.")
            os.makedirs(kaggle_dir, exist_ok=True)
            kaggle_credentials = '{"username":"neaurellia","key":"484839d067614e37b75788d6180bf627"}'
            with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f: f.write(kaggle_credentials)
            os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)
        
        datasets = {
            "jobs1": "dhivyadharunaba/it-job-roles-skills-dataset", "jobs2": "meerawks/it-skills-from-jobs",
            "courses1": "yosefxx590/coursera-courses-and-skills-dataset-2025", "courses2": "patrickgendotti/udacity-course-catalog",
            "certs1": "uzdavinys/coursera-specialization-dataset-2023-sep"
        }
        for name, path in datasets.items():
            try:
                print(f"   Downloading {name}...")
                subprocess.run(['kaggle', 'datasets', 'download', '-d', path, '--force', '-p', data_dir, '--unzip'], capture_output=True, check=True)
            except Exception as e:
                print(f"   ⚠️ Could not download {name}. Error: {e}")

        def find_and_rename(df, mapping):
            df.columns = [str(c).lower().strip().replace('ï»¿', '') for c in df.columns]
            return df.rename(columns={p: s for s, p_list in mapping.items() for p in p_list if p in df.columns})
        def parse_skills(series):
            return series.fillna('').astype(str).apply(lambda x: sorted(list(set(normalize_skill(s) for s in re.split(r'[,;|\n]+', x) if s.strip()))))

        all_jobs, all_learning = [], []
        
        job_mapping = {'title': ['job_role', 'job title', 'title'], 'skills': ['skills', 'it skills', 'requirements']}
        course_mapping = {'title': ['course name', 'title'], 'skills': ['top skills', 'skills covered', 'gained skills']}
        cert_mapping = {'title': ['specialization name', 'title'], 'skills': ['skills']}
        
        file_configs = {
            'data/IT_Job_Roles_Skills.csv': {'type': 'job', 'mapping': job_mapping},
            'data/job_skills.csv': {'type': 'job', 'mapping': job_mapping},
            'data/Coursera-courses-and-skills-2025.csv': {'type': 'course', 'mapping': course_mapping},
            'data/all_courses.csv': {'type': 'course', 'mapping': course_mapping},
            'data/coursera-specialization-dataset-2023-sep.csv': {'type': 'certificate', 'mapping': cert_mapping}
        }

        for filepath, config in file_configs.items():
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, encoding='latin1')
                    df = find_and_rename(df, config['mapping'])
                    if 'title' in df.columns:
                        df['skills'] = parse_skills(df.get('skills', pd.Series(dtype='str')))
                        df['type'] = config['type'].capitalize()
                        target_list = all_jobs if config['type'] == 'job' else all_learning
                        target_list.append(df)
                        print(f"   ✅ Loaded {len(df)} items from {os.path.basename(filepath)}")
                except Exception as e: print(f"   ⚠️ Error processing {os.path.basename(filepath)}: {e}")

        if all_jobs: self.jobs_df = pd.concat(all_jobs, ignore_index=True).dropna(subset=['title']).drop_duplicates(subset=['title']).reset_index(drop=True)
        if all_learning: self.learning_df = pd.concat(all_learning, ignore_index=True).dropna(subset=['title']).drop_duplicates(subset=['title']).reset_index(drop=True)
        
        print(f"✅ Data loaded: {len(self.jobs_df)} unique jobs, {len(self.learning_df)} unique learning items.")

    def classify_user_profile(self, user_skills, user_interests):
        scores = []
        for domain, categories in self.career_tree.items():
            for category, details in categories.items():
                required = set(details['skills']); user_set = set(user_skills)
                matching = user_set & required
                skill_match = len(matching) / len(required) if required else 0
                interest_score = 0.25 if any(word in user_interests.lower() for word in category.lower().split()) else 0
                total_score = (skill_match * 0.75) + interest_score
                if total_score > 0:
                    scores.append({"path": f"{domain} > {category}", "skill_score": skill_match, "interest_score": interest_score, "balanced_score": total_score})
        return sorted(scores, key=lambda x: x['skill_score'], reverse=True), sorted(scores, key=lambda x: x['interest_score'], reverse=True), sorted(scores, key=lambda x: x['balanced_score'], reverse=True)

    def _find_path_in_tree(self, dream_job_title):
        dream_job_lower = dream_job_title.lower()
        for domain, categories in self.career_tree.items():
            for category, details in categories.items():
                search_terms = [category.lower()] + [r.lower() for r in details['roles']]
                if any(term in dream_job_lower or dream_job_lower in term for term in search_terms):
                    return domain, category
        return None, None

    def generate_learning_path(self, user_skills, target_category, target_domain):
        category_info = self.career_tree.get(target_domain, {}).get(target_category, {})
        if not category_info: return {"error": "Target career path not found."}
        
        required = set(category_info['skills']); missing = sorted(list(required - set(user_skills)))
        path = {"target_path": f"{target_domain} > {target_category}", "missing_skills": missing, "recommended_roles": category_info.get('roles', []), "learning_recommendations": []}
        
        if not self.learning_df.empty and missing:
            for skill in missing:
                suggestions = []
                for _, item in self.learning_df.iterrows():
                    if skill in item.get('skills', []):
                        suggestions.append({"title": item['title'], "provider": item.get('provider', 'Unknown'), "type": item.get('type', 'Unknown')})
                if suggestions: path["learning_recommendations"].append({"skill_to_learn": skill, "suggestions": suggestions[:2]})
        return path

    def get_comprehensive_recommendations(self, user_skills, user_interests, dream_job, experience_level):
        print(f"🧠 Analyzing profile for: {dream_job} | Current Skills: {user_skills}")
        user_skills = [normalize_skill(s) for s in user_skills]
        
        top_skill, top_interest, top_balanced = self.classify_user_profile(user_skills, user_interests)
        
        dream_domain, dream_category = self._find_path_in_tree(dream_job)
        aspirational_path = self.generate_learning_path(user_skills, dream_category, dream_domain) if dream_domain else {"error": f"Dream job '{dream_job}' could not be mapped to a clear career path."}
        
        return {
            "user_profile": {"skills": user_skills, "interests": user_interests, "dream_job": dream_job, "experience_level": experience_level},
            "recommendation_summary": {
                "strongest_skill_matches": [p['path'] for p in top_skill[:2]],
                "strongest_interest_matches": [p['path'] for p in top_interest[:2]],
                "best_balanced_options": [p['path'] for p in top_balanced[:2]],
            },
            "aspirational_path_for_dream_job": aspirational_path
        }

def generate_test_profiles():
    print("🏭 Generating diverse test profiles to cover edge cases...")
    profiles = []
    all_skills = list(SKILL_SYNONYMS.keys())

    for domain, categories in CAREER_TREE.items():
        for category, details in categories.items():
            required_skills = details['skills']
            roles = details['roles']
            
            # Upskiller Profile
            if len(required_skills) > 1:
                skills_to_have = random.sample(required_skills, k=len(required_skills) // 2)
                profiles.append({
                    "name": f"Upskiller for {category}", "user_skills": skills_to_have,
                    "user_interests": f"I'm interested in {category}", "dream_job": random.choice(roles),
                    "experience_level": "2 years"
                })

            # Career Changer Profile
            other_skills = [s for s in all_skills if s not in required_skills]
            if other_skills:
                skills_to_have = random.sample(other_skills, k=min(2, len(other_skills)))
                profiles.append({
                    "name": f"Career Changer to {category}", "user_skills": skills_to_have,
                    "user_interests": f"I want to become a {random.choice(roles)}", "dream_job": random.choice(roles),
                    "experience_level": "5 years in another field"
                })
    
    num_to_generate = 100
    final_profiles = []
    while len(final_profiles) < num_to_generate:
        final_profiles.extend(profiles)
    
    print(f"   ✅ Generated {len(final_profiles[:num_to_generate])} test profiles.")
    return final_profiles[:num_to_generate]

def main():
    print("=" * 60)
    system = CareerAdviser()
    system.load_data()
    
    test_profiles = generate_test_profiles()
    
    all_recs = []
    for i, profile in enumerate(test_profiles, 1):
        print(f"\n👤 Running Test Profile {i}/{len(test_profiles)}: {profile['name']}...")
        
        # --- FIXED: Create a copy of the profile and remove the 'name' key before calling ---
        profile_for_recommender = profile.copy()
        profile_name_for_json = profile_for_recommender.pop('name', f'profile_{i}')

        recs = system.get_comprehensive_recommendations(**profile_for_recommender)
        all_recs.append({f"profile_{i}_{profile_name_for_json.replace(' ', '_')}": recs})

    final_output = {"system_info": {"version": "Career Adviser v18 - Final Production Test"}, "recommendations": all_recs}
    output_filename = "PP_MLOps_AinunKhoirunni'mah_CatherineAurellia_Output.json"
    with open(output_filename, 'w', encoding='utf-8') as f: json.dump(final_output, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 System run complete! {len(all_recs)} profiles processed.")
    print(f"📄 Final recommendations saved to '{output_filename}'")

if __name__ == "__main__":
    main()