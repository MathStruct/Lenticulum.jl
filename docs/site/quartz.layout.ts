import { siteLayout } from "./mathstruct/layout"

// The footer links back to the API documentation (the Documenter site under api/) and to the
// repository. Every note has a giscus comment thread, stored as a GitHub Discussion in the
// repository's "Announcements" category (only maintainers and the giscus app open threads
// there; anyone signed in to GitHub can reply and react). A note opts out with
// `comments: false` in its frontmatter.
export const { sharedPageComponents, defaultContentPageLayout, defaultListPageLayout } =
  siteLayout({
    footerLinks: {
      // set by docs/make.jl through docs/site/build.sh: api/dev/ on master, api/preview/ on a preview
      "API documentation":
        process.env.VAULT_API_URL ?? "https://mathstruct.org/Lenticulum.jl/api/dev/",
      GitHub: "https://github.com/MathStruct/Lenticulum.jl",
    },
    comments: {
      repo: "MathStruct/Lenticulum.jl",
      repoId: "R_kgDOUPo0Lw",
      category: "Announcements",
      categoryId: "DIC_kwDOUPo0L84DIUJp",
    },
    // no threads on a preview build (VAULT_COMMENTS=0, set by docs/make.jl off `master`)
    commentsEnabled: process.env.VAULT_COMMENTS !== "0",
  })
