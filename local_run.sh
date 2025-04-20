/usr/local/lib/ruby/gems/3.3.0/bin/jekyll new blogs
/usr/local/lib/ruby/gems/3.3.0/bin/bundle exec jekyll serve --livereload

/usr/local/lib/ruby/gems/3.3.0/bin/bundle add jekyll-katex
/usr/local/lib/ruby/gems/3.3.0/bin/bundle add jekyll-include-cache

### img fixes
#       <img src="{{ site.baseurl }}/assets/imgs/..." width="50%" height="50%" alt="..." />

### Katex Fixes (only if they are in $...$)
# next "_" be replaced with "\_", in particular, do "}_{" replaced with "}\_{"
# replace all "|" with "\mid"