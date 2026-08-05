import { Link } from "react-router-dom";
import LegalPage, { H2, P, Table } from "../components/LegalPage";

/**
 * What actually happens to someone's documents.
 *
 * Written from the code rather than from a template, and deliberately specific:
 * a policy that says "we may share data with partners" when there are no
 * partners is worse than no policy, because it trains people to ignore the one
 * they should read. Every claim here corresponds to something in the repo, and
 * the operator notes say plainly where it depends on how this instance is run.
 */
export default function Privacy() {
  return (
    <LegalPage title="How your data is handled" updated="August 2026">
      <P>
        Scholar answers questions using only documents you upload. This page describes
        what that means for those documents — where they go, who can read them, and
        how to get them back or destroy them.
      </P>

      <H2>What is stored</H2>
      <Table
        head={["What", "Why", "Where"]}
        rows={[
          ["Your email address", "identifies the account; used for password resets and workspace invitations", "this server's database"],
          ["Your password", "stored only as a bcrypt hash — it cannot be read back or recovered, only reset", "this server's database"],
          ["The documents you upload", "served back to you when you open a citation", "this server's disk"],
          ["Extracted text, chunks and embeddings", "this is what makes the documents searchable", "this server's disk"],
          ["Questions, answers and their evidence", "the audit trail — so an answer can be checked later against the passages that produced it", "this server's database"],
        ]}
      />

      <H2>Who can read your documents</H2>
      <P>
        You, and anyone who is a member of a workspace you put them in. Nobody else.
        Your personal library cannot be shared at all — that is enforced in the code,
        not by policy. A request for a workspace you are not a member of is answered
        as though it does not exist.
      </P>
      <P>
        Whoever operates this server has administrative access to the machine, and
        therefore to the files on it. That is true of every hosted service; it is
        stated here because it is the honest answer, not because it is unusual.
      </P>

      <H2>What is sent to third parties</H2>
      <P>
        <strong>Your documents are not sent to any third-party AI provider.</strong>{" "}
        Retrieval, ranking and answer generation all run on this server using models
        that run here. Your documents are not used to train any model, ours or
        anyone else's.
      </P>
      <P>
        The one exception is email. If this instance is configured to send mail,
        password-reset and workspace-invitation messages pass through that mail
        provider. Those messages contain a link and the recipient's address —
        never your documents or your answers.
      </P>
      <P>
        There are no analytics, no advertising trackers, and no third-party scripts.
      </P>

      <H2>Cookies</H2>
      <P>
        Three, all strictly necessary — there is nothing here to opt out of because
        nothing is optional. Two are <code>httpOnly</code>, meaning JavaScript cannot
        read them: a short-lived session token and a longer-lived refresh token that
        is only sent to the one endpoint that renews sessions. The third is a CSRF
        token, which must be readable by the page in order to do its job.
      </P>

      <H2>Getting your data out, and deleting it</H2>
      <P>
        <strong>Export</strong> gives you a <code>.tar.gz</code> containing your
        account details, your workspaces, every question and answer with its full
        evidence chain, and the original files you uploaded — the actual files, not a
        list of them. Documents uploaded by other people into shared workspaces are
        not included, because they are not yours to take.
      </P>
      <P>
        <strong>Deletion</strong> removes the account, its libraries, the extracted
        text and embeddings, the audit log, and the uploaded files themselves from
        disk. It is immediate and cannot be undone. If you are the last owner of a
        shared workspace that other people are still using, deletion is refused until
        you make someone else an owner — those documents belong to the workspace, and
        destroying a team's library as a side effect of closing your account would be
        the wrong default.
      </P>
      <P>
        Both are in <Link to="/app" className="text-accent underline underline-offset-2">your
        account settings</Link>.
      </P>

      <H2>How long things are kept</H2>
      <P>
        Until you delete them. There is no automatic expiry of documents or answers.
        Expired login tokens are cleared automatically; password-reset links stop
        working after 30 minutes and can only be used once.
      </P>

      <H2>If you run this yourself</H2>
      <P>
        Scholar is open source and self-hostable. Running your own instance means the
        documents never leave infrastructure you control, and nothing above involves
        anyone but you.
      </P>

      <div className="mt-10 rounded-[10px] border border-line bg-panel px-5 py-4">
        <p className="font-mono text-[0.62rem] uppercase tracking-[0.16em] text-faint">
          note for whoever deploys this
        </p>
        <p className="mt-2 text-[0.88rem] leading-relaxed text-graphite">
          This page describes what the software does, accurately, as of the version
          you are running. It is <strong>not legal advice and is not a complete
          privacy policy</strong> for a public service — that needs your identity as
          the data controller, a contact address, your hosting jurisdiction and
          sub-processors, a lawful basis if you serve users in the EU or UK, and a
          lawyer's eyes. Fill those in before taking payment or advertising this to
          the public.
        </p>
      </div>
    </LegalPage>
  );
}
