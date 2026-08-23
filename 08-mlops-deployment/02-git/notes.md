# 02 – Git & GitHub

## Problem

Code (and, in this course, the notes/notebooks themselves) changes constantly, often by more than one person, and across time. Two problems compound: **coordination** — how do two people work on the same files without silently overwriting each other's changes — and **history** — how do you get back to (or even just understand) a working state from three weeks ago, once today's version has diverged from it in ways nobody wrote down.

**How do we track every change to a set of files over time, safely combine changes made independently by different people, and always be able to recover any previous state — without a human manually keeping track of any of it?**

## Intuition

Imagine an old-fashioned approach: every time you make a meaningful change to a script, you save a new copy — `train.py`, `train_v2.py`, `train_v2_final.py`, `train_v2_final_ACTUALLY_FINAL.py`. This "solves" the history problem in the crudest possible way (every version technically still exists, somewhere), but it's obviously unworkable the moment more than one person is involved, or the project runs longer than an afternoon.

Git's actual approach is closer to how a photographer thinks about a photo library, not how a word processor thinks about "track changes": every commit is a **complete, addressable snapshot** of the whole project at one moment, and snapshots are linked together into a history graph. Crucially — and this is the part that makes Git fast and reliable rather than just another version-numbering scheme — a snapshot is not identified by a filename or a sequence number a human assigns. It's identified by the **hash of its own content**. Two snapshots (or two files, or two folders) with identical content are, as far as Git is concerned, *the same object*, no matter when or by whom they were created. This single idea — content-addressing — is what lets Git deduplicate, verify integrity, and combine independent lines of work reliably, as the rest of this topic works out in detail.

## Why simpler approaches fail

The numbered-copy approach (`script_v2_final_FINAL.py`) fails for reasons that are structural, not just aesthetic:

1. **It doesn't merge.** If two people each independently improve `train.py` — one adds logging, the other tunes a hyperparameter — numbered copies give you two full files with no way to combine "the logging change" and "the hyperparameter change" into one file without manually re-typing one person's edit into the other's copy by eye. There's no notion of *which lines* changed, only *entire alternate versions* of the whole file.
2. **It doesn't show who changed what, or why.** A filename like `train_v2_final.py` records that a change happened and (loosely) an ordering, but nothing about *what specifically* changed between v1 and v2, who made the change, or the reasoning behind it. Debugging "when did this bug get introduced" turns into manually diffing entire files by eye across however many numbered copies exist.
3. **It doesn't let you cheaply branch to try something risky.** Trying a risky experimental change under the numbered-copy scheme means either overwriting the only working copy (losing the ability to go back) or making yet another full copy of the entire project — expensive, and now you have to remember to reconcile it later, by hand, if the experiment works out.

Git solves all three by tracking changes at the level of individual, hashed, addressable content, and by making "try something risky in isolation" (branching) and "combine two independent lines of work" (merging) cheap, structural operations instead of manual, error-prone ones.

## Conceptual foundation

*(Systems topic — no calculus-style mathematics; the real depth here is Git's object model, per AGENTS.md's documented Math→Conceptual-foundation substitution for pure-systems topics.)*

### Git's object model: a content-addressable DAG

Every piece of data Git tracks is stored as an **object**, and every object's identity is the cryptographic hash (SHA-1 historically, SHA-256 in newer repositories) of its own content. There are three object types relevant here:

- **Blob** — the raw content of one file, with no filename or metadata attached. Two files with byte-identical content, anywhere in the project, at any point in history, are the *same* blob, stored once.
- **Tree** — a directory listing: a set of (name, mode, object-hash) entries, each pointing at either a blob (a file) or another tree (a subdirectory). A tree is Git's snapshot of one directory's structure at one moment.
- **Commit** — a pointer to one root tree (the snapshot of the entire project at that moment), plus metadata (author, message, timestamp) and pointers to the commit(s) that came immediately before it (its "parent(s)").

Because every object is identified purely by a hash of its content, and every commit points to its parent(s) by hash, the full history of a repository is a **Directed Acyclic Graph (DAG)** of content-addressed objects — "acyclic" because a commit can never point back to a later commit as its own ancestor; history only ever extends forward. Two commits with completely different histories that happen to produce the exact same file content in some directory transparently share that content's blob object; Git never notices or cares that the two histories are otherwise unrelated.

**This is exactly the same idea as `01-docker/notes.md`'s "Conceptual foundation" section** — Docker image layers are content-addressed diffs, chained together, with identical layers shared and deduplicated automatically. Git's blobs/trees/commits are content-addressed *snapshots*, chained together the same way. Both systems get the same three properties for free from the same underlying idea: deduplication (identical content is stored once, no matter where it recurs), integrity verification (re-hash the content, compare to its own address — if they don't match, the data was corrupted or tampered with), and efficient "what changed" comparisons (comparing two hashes is instant; you don't need to compare full content unless the hashes actually differ).

### Branches are just movable pointers

A **branch** (e.g. `main`, `feature-lr-001`) is not a copy of anything — it is a single, tiny, mutable pointer to one commit (the branch's current "tip"). Creating a branch is cheap (writing one new pointer, not copying any files); committing on a branch moves that branch's pointer forward to the new commit; switching branches (`git checkout`) just updates which commit your working directory reflects. This is *why* branching in Git is cheap enough to use constantly for even small, throwaway experiments, unlike the "make another full copy of the project" alternative from "Why simpler approaches fail."

## From-scratch implementation

`mini_blob_store.py` (in this folder) implements, in plain Python, the single primitive every Git object above is built from: hash content, store it under its own hash as the filename, retrieve it later purely by that hash. (Git additionally prefixes content with a type+size header before hashing, and zlib-compresses the stored bytes — both omitted here to keep the core mechanism visible, and noted rather than reimplemented, since neither changes the essential idea.)

Actually executed with `.venv/bin/python mini_blob_store.py`; real captured output:

```
=== Storing two different blobs ===
blob 1 hash: d6a06ca97d44b06fa524cc949578bbf1028cd58386543c0cad705c79f00a7759
blob 2 hash: 9cef0ea5e4e18c238e164d1c89e57bde18389cdbe00198f812537a51ddaf94dc
hashes differ (different content): True

=== Storing the SAME content twice (deduplication) ===
blob 3 hash: d6a06ca97d44b06fa524cc949578bbf1028cd58386543c0cad705c79f00a7759
h1 == h3 (identical content -> identical, deduped address): True
objects actually on disk (2): ['9cef0ea5e4e18c238e164d1c89e57bde18389cdbe00198f812537a51ddaf94dc', 'd6a06ca97d44b06fa524cc949578bbf1028cd58386543c0cad705c79f00a7759']

=== Retrieving by hash alone ===
retrieve(h1) ->
'def train_model():\n    return model.fit(X, y)\n'

=== Changing content by even one character changes the address ===
blob 4 hash (no trailing newline): c27835564d78fba0580fb5ee67f90d006001f34c4a3e36f53e4cf6678b787bd1
h1 != h4 (one-byte diff -> completely different hash): True

=== Integrity check ===
verify_integrity(h1): True
```

Note in particular: `store()` was called three times with only two distinct pieces of content, and only **2 objects** ended up on disk (`h1 == h3`) — storing identical content twice is a genuine no-op, which is exactly why Git's actual object store dedupes identical file content across every commit in a repository's entire history for free, with no extra bookkeeping required.

## Practical implementation

A real `git init` → `add` → `commit` → `log` walkthrough, actually run in a scratch directory (`/tmp/git-demo`, outside this repository's own history) with `git version 2.43.0`. Real captured terminal output:

```
$ git init
Initialized empty Git repository in /tmp/git-demo/.git/

$ git config user.email "demo@example.com"; git config user.name "Demo User"

$ echo "print('training model v1')" > train.py
$ git add train.py
$ git status
On branch main

No commits yet

Changes to be committed:
  (use "git rm --cached <file>..." to unstage)
	new file:   train.py


$ git commit -m "Initial commit: add train.py"
[main (root-commit) f3843b7] Initial commit: add train.py
 1 file changed, 1 insertion(+)
 create mode 100644 train.py

$ git log --oneline
f3843b7 Initial commit: add train.py
```

### A real merge conflict, deliberately created and resolved

Two branches independently edit the *same line* of `train.py` (a classic, unavoidable conflict — line-level, not file-level, so Git's automatic 3-way merge cannot pick a side on its own):

```
$ git checkout -b feature-lr-001
Switched to a new branch 'feature-lr-001'
$ sed -i "s/v1/v2-lr-0.01/" train.py; cat train.py
print('training model v2-lr-0.01')
$ git commit -am "feature branch: bump learning rate to 0.01"
[feature-lr-001 b203e76] feature branch: bump learning rate to 0.01
 1 file changed, 1 insertion(+), 1 deletion(-)

$ git checkout main
Switched to branch 'main'
$ sed -i "s/v1/v2-lr-0.1/" train.py; cat train.py
print('training model v2-lr-0.1')
$ git commit -am "main branch: bump learning rate to 0.1"
[main 070b070] main branch: bump learning rate to 0.1
 1 file changed, 1 insertion(+), 1 deletion(-)

$ git merge feature-lr-001
Auto-merging train.py
CONFLICT (content): Merge conflict in train.py
Automatic merge failed; fix conflicts and then commit the result.

$ git status
On branch main
You have unmerged paths.
  (fix conflicts and run "git commit")
  (use "git merge --abort" to abort the merge)

Unmerged paths:
  (use "git add <file>..." to mark resolution)
	both modified:   train.py

no changes added to commit (use "git add" and/or "git commit -a")

$ cat train.py   # conflict markers
<<<<<<< HEAD
print('training model v2-lr-0.1')
=======
print('training model v2-lr-0.01')
>>>>>>> feature-lr-001
```

Resolving by hand (a human decision Git cannot make on its own — here, keeping the feature branch's value after review) and completing the merge:

```
$ cat train.py
print('training model v2-lr-0.01')  # resolved: keep feature branch value after review

$ git add train.py
$ git commit -m "Merge feature-lr-001: resolve learning-rate conflict, keep 0.01"
[main 9f76957] Merge feature-lr-001: resolve learning-rate conflict, keep 0.01

$ git log --oneline --graph --all
*   9f76957 Merge feature-lr-001: resolve learning-rate conflict, keep 0.01
|\  
| * b203e76 feature branch: bump learning rate to 0.01
* | 070b070 main branch: bump learning rate to 0.1
|/  
* f3843b7 Initial commit: add train.py

$ git status
On branch main
nothing to commit, working tree clean
```

The `git log --graph` output makes the DAG structure from "Conceptual foundation" literal and visible: `f3843b7` is the single common ancestor; `070b070` and `b203e76` are two commits that both point back to it as their parent (a fork); `9f76957` is a merge commit with **two** parents, joining the graph back into one line.

## Experiment

**Hypothesis:** a 3-way merge (comparing the common ancestor, `main`'s tip, and `feature-lr-001`'s tip) auto-resolves cleanly whenever the two branches touch *different* lines, but must stop and ask a human whenever both branches change the *same* line to *different* values, because Git has no way to know which edit should "win" (or whether both should be combined, and how).

**Setup:** exactly the walkthrough above — both branches modify the single line in `train.py` that sets the learning rate, to two different values (`0.1` on `main`, `0.01` on `feature-lr-001`), starting from the same common-ancestor commit `f3843b7`.

**Actual result:** `git merge feature-lr-001` printed `CONFLICT (content): Merge conflict in train.py` and left conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) directly in the file, exactly as hypothesized — Git detected that the same line diverged from the common ancestor in two incompatible directions and refused to guess.

**Interpretation:** this confirms Git's 3-way merge algorithm operates at line granularity, not file granularity — had the two branches edited *different* lines of `train.py` (e.g. one adding a new function at the bottom while the other changed an unrelated line near the top), `git merge` would have silently and successfully combined both changes into one file with no conflict and no human intervention needed. Conflicts are not a sign that something went wrong with Git; they are the expected, correct outcome whenever two people's actual edits are genuinely incompatible and no algorithm can know which was intended.

**Limitations:** this experiment used the simplest possible conflict (one line, two branches, one file). Real conflicts often involve multiple overlapping hunks, renamed/deleted files, or three or more diverging branches, which more advanced tools (`git rerere`, semantic/structured merge tools) exist to make less painful — none of that is exercised here.

## Failure modes

- **Force-pushing over shared history.** `git push --force` rewrites the remote branch's history to match the local branch exactly, discarding any commits on the remote that aren't in the local history — including commits a teammate already pushed and is building on. Anyone who had already pulled the old history now has a branch that has silently diverged from the remote in a way `git pull` won't cleanly reconcile. `git push --force-with-lease` (which refuses to overwrite the remote unless it still points where you last saw it) is the safer default when a force-push is genuinely necessary.
- **Committing secrets.** An API key or credential committed to a repository is not "removed" by a later commit that deletes it — per "Conceptual foundation," each commit's tree/blob objects are immutable snapshots addressed by content, and the secret's blob object still exists, still reachable from the earlier commit, for as long as that commit is reachable in history (which, once pushed, may be forever, on every clone). Removing a leaked secret requires rewriting history (`git filter-repo` or similar) and treating the secret itself as compromised and rotated — deleting the file in a new commit is not sufficient.
- **Merge conflicts from long-lived branches.** The longer a branch lives without merging back to `main` (or being merged from `main`), the more both sides accumulate independent changes to overlapping code, and the larger and harder-to-resolve the eventual conflict becomes. This is a direct argument for frequent, small merges over large, infrequent ones — a habit, not a Git feature.

## Real-world usage

- **Every collaborative codebase** (including this course's own repository) uses Git as its foundation for exactly the coordination + history problem stated at the top of this topic.
- **CI/CD pipelines** (see `08-mlops-deployment`'s later `07-cicd` topic) trigger off Git events (a push, a pull request) — the entire automation model of modern software delivery is built on top of Git's commit graph as the source of truth for "what changed."
- **Model/data versioning tools** (DVC, in this section's `03`/`04` topics) extend Git's content-addressing idea to large binary artifacts (datasets, model weights) that don't belong directly in a Git repository, using the same core principle: content is identified by its hash, not by a mutable filename.

## Mental model

**A Git repository is a content-addressed DAG of immutable snapshots; a branch is just a movable pointer into that graph.** Committing extends the graph forward; branching adds a cheap new pointer; merging joins two divergent paths back into one, automatically wherever the underlying changes don't overlap, and asking a human only at the exact lines where they genuinely conflict.

## Questions to think about

1. `mini_blob_store.py`'s Experiment showed that storing identical content twice produces the same hash and no duplicate file on disk. If a 50 MB dataset file is committed to a Git repository, and then committed again completely unchanged in ten later commits, how much additional storage does Git actually use for that file across those ten later commits — and why?
2. In the merge-conflict walkthrough, Git detected a conflict because both branches changed the *same line*. If instead `main` had deleted `train.py` entirely while `feature-lr-001` edited a line inside it, would you expect Git to auto-resolve that, or treat it as a conflict? What kind of conflict marker or message would make sense for that case?
3. `01-docker/notes.md`'s Dockerfile pins its base image to an exact tag rather than `latest`, for reproducibility. What is the Git equivalent of "pinning" a dependency to an exact, reproducible reference rather than a moving one (hint: think about what a branch name like `main` points to versus what a specific commit hash points to)?
4. Two teammates both branch off the same commit and, days later, both need to merge back to `main`. One merges first with no conflicts. Why might the *second* teammate's merge now conflict on lines they never touched themselves, even though their own changes didn't overlap with the first teammate's changes when they originally branched?
