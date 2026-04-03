# GitHub SSH setup

Run these commands locally:

```bash
ssh-keygen -t ed25519 -C "YOUR_GITHUB_EMAIL"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
cat ~/.ssh/id_ed25519.pub
```

Then add the public key to GitHub and test:

```bash
ssh -T git@github.com
```

Recommended remote workflow:

```bash
cd ~/src/vericodec-diff
git remote add origin git@github.com:raddshing/vericodec-diff.git
git add .
git commit -m "bootstrap: initialize repo skeleton"
git push -u origin main
```
