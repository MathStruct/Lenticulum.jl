import { PageLayout, SharedLayout } from "./quartz/cfg"
import * as Component from "./quartz/components"

// Copied over the stock `quartz.layout.ts` by `docs/site/build.sh`.
// Deliberate changes from stock: the footer links back to the API documentation (the
// Documenter site under api/) and to the repository, and every note has a
// Giscus comment thread, stored as a GitHub Discussion in the repository's "Announcements"
// category (only maintainers and the giscus app open threads there; anyone signed in to GitHub
// can reply and react). A note opts out with `comments: false` in its frontmatter.

export const sharedPageComponents: SharedLayout = {
  head: Component.Head(),
  header: [],
  afterBody: [
    Component.ConditionalRender({
      component: Component.Comments({
        provider: "giscus",
        options: {
          repo: "MathStruct/Lenticulum.jl",
          repoId: "R_kgDOUPo0Lw",
          category: "Announcements",
          categoryId: "DIC_kwDOUPo0L84DIUJp",
          mapping: "pathname",
          strict: true,
          reactionsEnabled: true,
          inputPosition: "bottom",
        },
      }),
      // no threads on a preview build (VAULT_COMMENTS=0, set by docs/make.jl off `master`)
      condition: (page) =>
        process.env.VAULT_COMMENTS !== "0" && !(page.fileData.slug ?? "").startsWith("tags/"),
    }),
  ],
  footer: Component.Footer({
    links: {
      // set by docs/make.jl through docs/site/build.sh: api/dev/ on master, api/preview/ on a preview
      "API documentation": process.env.VAULT_API_URL ?? "https://mathstruct.org/Lenticulum.jl/api/dev/",
      GitHub: "https://github.com/MathStruct/Lenticulum.jl",
    },
  }),
}

export const defaultContentPageLayout: PageLayout = {
  beforeBody: [
    Component.ConditionalRender({
      component: Component.Breadcrumbs(),
      condition: (page) => page.fileData.slug !== "index",
    }),
    Component.ArticleTitle(),
    Component.ContentMeta(),
    Component.TagList(),
  ],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        { Component: Component.Search(), grow: true },
        { Component: Component.Darkmode() },
        { Component: Component.ReaderMode() },
      ],
    }),
    Component.Explorer(),
  ],
  right: [
    Component.Graph(),
    Component.DesktopOnly(Component.TableOfContents()),
    Component.Backlinks(),
  ],
}

export const defaultListPageLayout: PageLayout = {
  beforeBody: [Component.Breadcrumbs(), Component.ArticleTitle(), Component.ContentMeta()],
  left: [
    Component.PageTitle(),
    Component.MobileOnly(Component.Spacer()),
    Component.Flex({
      components: [
        { Component: Component.Search(), grow: true },
        { Component: Component.Darkmode() },
      ],
    }),
    Component.Explorer(),
  ],
  right: [],
}
