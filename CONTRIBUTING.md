# git, Jupyter notebooks, and `nbstripout`

To keep the repository clean, we use `nbstripout` to remove Jupyter Notebook cell outputs. Below are instructions to install and configure the `nbstripout` filter. Our `.gitattributes` file relies on this tool to automatically strip outputs when you stage a notebook for commit.

## Install and Configure nbstripout

### Step 1: Install nbstripout

Use the installation method that works best for your environment:

<table>
<thead>
<tr>
<th align="left">Tool</th>
<th align="left">Installation Command</th>
<th align="left">Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td align="left"><strong>Conda</strong></td>
<td align="left"><code>conda install -c conda-forge nbstripout</code></td>
<td align="left">Recommended if using a Conda environment.</td>
</tr>
<tr>
<td align="left"><strong>pipx (Recommended CLI)</strong></td>
<td align="left"><code>pipx install nbstripout</code></td>
<td align="left">Requires <code>pip install pipx</code> first. Installs <code>nbstripout</code> isolated from project dependencies.</td>
</tr>
<tr>
<td align="left"><strong>pip (Standard)</strong></td>
<td align="left"><code>pip install nbstripout</code></td>
<td align="left">Standard Python package installation.</td>
</tr>
</tbody>
</table>

### Step 2: Configure Git

Once nbstripout is installed, run this single command to tell Git to use the filter globally for all projects that have a compatible `.gitattributes` file:
```
nbstripout --install
```

Alternatively, run
```
git config --global filter.nbstripout.clean "nbstripout"
git config --global filter.nbstripout.smudge "cat"
```
from the Terminal.

## Verification

You can verify the setup by running:
```
git config --global --get filter.nbstripout.clean
```

The output should be `nbstripout`. If it is, your environment is correctly configured to contribute clean notebook files!
