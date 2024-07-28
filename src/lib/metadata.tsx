import React from "react";
import { Helmet } from "react-helmet";

interface MetadataProps {
  title?: string;
  description?: string;
  keywords?: string;
  author?: string;
  ogTitle?: string;
  ogDescription?: string;
  ogImage?: string;
  ogUrl?: string;
  twitterCard?: string;
  twitterTitle?: string;
  twitterDescription?: string;
  twitterImage?: string;
  twitterSite?: string;
  favicon?: string;
}

const defaultMetadata = {
  title: "SIBI Sign App",
  description:
    "A comprehensive app for learning SIBI (Sistem Isyarat Bahasa Indonesia).",
  keywords: "SIBI, sign language, learning, app, Indonesian Sign Language",
  author: "name",
  ogTitle: "SIBI Sign App",
  ogDescription:
    "A comprehensive app for learning SIBI (Sistem Isyarat Bahasa Indonesia).",
  ogImage: "/logo.png",
  ogUrl: "https://apapun.com",
  twitterCard: "summary_large_image",
  twitterTitle: "SIBI Sign App",
  twitterDescription:
    "A comprehensive app for learning SIBI (Sistem Isyarat Bahasa Indonesia).",
  twitterImage: "/logo.png",
  twitterSite: "@yourtwitterhandle",
  favicon: "/logo.png",
};

const Metadata: React.FC<MetadataProps> = ({
  title = defaultMetadata.title,
  description = defaultMetadata.description,
  keywords = defaultMetadata.keywords,
  author = defaultMetadata.author,
  ogTitle = defaultMetadata.ogTitle,
  ogDescription = defaultMetadata.ogDescription,
  ogImage = defaultMetadata.ogImage,
  ogUrl = defaultMetadata.ogUrl,
  twitterCard = defaultMetadata.twitterCard,
  twitterTitle = defaultMetadata.twitterTitle,
  twitterDescription = defaultMetadata.twitterDescription,
  twitterImage = defaultMetadata.twitterImage,
  twitterSite = defaultMetadata.twitterSite,
  favicon = defaultMetadata.favicon,
}) => {
  return (
    <Helmet>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <title>{title}</title>
      <link rel="icon" type="image/png" href={favicon} />
      <meta name="description" content={description} />
      <meta name="keywords" content={keywords} />
      <meta name="author" content={author} />
      <meta property="og:title" content={ogTitle} />
      <meta property="og:description" content={ogDescription} />
      <meta property="og:image" content={ogImage} />
      <meta property="og:url" content={ogUrl} />
      <meta property="og:type" content="website" />
      <meta name="twitter:card" content={twitterCard} />
      <meta name="twitter:title" content={twitterTitle} />
      <meta name="twitter:description" content={twitterDescription} />
      <meta name="twitter:image" content={twitterImage} />
      <meta name="twitter:site" content={twitterSite} />
    </Helmet>
  );
};

export default Metadata;
