import { Link } from "react-router-dom";
import LegalPage, { H2, P } from "../components/LegalPage";

/**
 * Terms of use.
 *
 * Kept short and readable on purpose. The one clause that matters for this
 * product in particular is the accuracy one: an AI that cites its sources
 * invites more trust than one that doesn't, so the limits have to be stated
 * where people will actually read them.
 */
export default function Terms() {
  return (
    <LegalPage title="Terms of use" updated="August 2026">
      <P>
        Plain English, because terms nobody reads protect nobody.
      </P>

      <H2>What Scholar does</H2>
      <P>
        You upload documents. Scholar finds relevant passages in them and uses a
        language model to write an answer from those passages, citing each one. It
        answers from your documents, not from the model's general knowledge.
      </P>

      <H2>Accuracy — read this one</H2>
      <P>
        <strong>Scholar can be wrong.</strong> Retrieval can miss the passage that
        actually answers your question, and the language model can misread a passage
        it was given. Citations tell you where an answer came from; they do not
        guarantee the answer is a correct reading of the source.
      </P>
      <P>
        Every claim links to its source so you can check it. For anything that
        matters — a legal, financial, medical or safety decision — read the cited
        passage. Scholar is a tool for finding and quoting evidence in your own
        documents, not a substitute for professional advice or for reading the
        document yourself.
      </P>

      <H2>Your documents</H2>
      <P>
        They remain yours. Uploading grants only the permission needed to run the
        service: storing the files, extracting and indexing their text, and showing
        passages back to you and to members of workspaces you share them with. They
        are not used to train models. See{" "}
        <Link to="/privacy" className="text-accent underline underline-offset-2">
          how your data is handled
        </Link>.
      </P>
      <P>
        Only upload documents you have the right to upload. If you put someone else's
        confidential material into a shared workspace, that is your call and your
        responsibility.
      </P>

      <H2>Acceptable use</H2>
      <P>
        Don't use Scholar to break the law, to infringe someone's copyright or
        privacy, or to attack the service — probing other accounts, scraping,
        or deliberately overloading it. Usage limits exist and are enforced per
        account.
      </P>

      <H2>Availability</H2>
      <P>
        No uptime is promised. Keep your own copies of anything important; you can
        export everything at any time.
      </P>

      <H2>Ending it</H2>
      <P>
        Delete your account whenever you like — it removes your documents and data
        from the server immediately and permanently.
      </P>

      <div className="mt-10 rounded-[10px] border border-line bg-panel px-5 py-4">
        <p className="font-mono text-[0.62rem] uppercase tracking-[0.16em] text-faint">
          note for whoever deploys this
        </p>
        <p className="mt-2 text-[0.88rem] leading-relaxed text-graphite">
          This is a readable starting point, <strong>not a lawyer-reviewed contract</strong>.
          Before running this as a public or paid service you need to add the operating
          entity, a governing law and jurisdiction, liability and warranty clauses
          appropriate to it, and a contact address — and have someone qualified read
          the result.
        </p>
      </div>
    </LegalPage>
  );
}
