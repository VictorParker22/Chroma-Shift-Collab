---
description: Commit changes and push to GitHub
---

# Push to GitHub

// turbo-all

## Steps

1. Check status:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
git status
```

2. Stage all changes:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
git add -A
```

3. Commit with a descriptive message:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
git commit -m "descriptive commit message here"
```

4. Push to main:
```bash
cd /home/chroma-shift/Documents/Chroma-Shift
git push
```

## Notes
- Repo: github.com/VictorParker22/Chroma-Shift
- Branch: main
- .gitignore excludes: models/vila/, *.engine, .env, __pycache__
- Always write a meaningful commit message describing WHAT changed and WHY
