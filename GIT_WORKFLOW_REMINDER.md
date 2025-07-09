# 🚨 CRITICAL GIT WORKFLOW REMINDER 🚨

## **IMPORTANT: Changes Must Be Pushed to Reach VPS**

### **The Problem:**
- I make changes locally on Mac/Cursor
- User expects them on Ubuntu VPS
- **IF NOT PUSHED → CHANGES DON'T EXIST ON VPS**

### **The Workflow:**
1. **Make changes locally** (Cursor/Mac)
2. **Commit changes** (`git add . && git commit -m "description"`)
3. **Push to GitHub** (`git push origin main`)
4. **VPS pulls changes** (`git pull origin main`)

### **Before Expecting Changes on VPS:**
✅ **ALWAYS CHECK:** Have I pushed the changes?
```bash
# Check if changes are committed and pushed
git status
git log --oneline -5
```

### **VPS Commands to Get Latest:**
```bash
# On Ubuntu VPS
cd ~/aai
git pull origin main
# Then run the updated scripts
```

### **Common Mistakes:**
- ❌ Making local changes and expecting them on VPS immediately
- ❌ Forgetting to commit/push before switching contexts
- ❌ User runs old version of script on VPS

### **Quick Check:**
- **Local**: `git status` (should be clean if all pushed)
- **VPS**: `git pull origin main` (should say "Already up to date" if synced)

---
**REMEMBER: No push = No VPS deployment!** 