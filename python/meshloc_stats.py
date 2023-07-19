from matplotlib import pyplot as plt
from absl import app
from absl import flags
from pathlib import Path
import numpy as np
import scipy
import pandas as pd
import seaborn as sns
import collections


FLAGS = flags.FLAGS
flags.DEFINE_string(
    "colmap_image_txt",
    None,
    "The path to the images.txt of the colmap reconstruction. The reconstruction need to be transformed to the mesh coodinate",
)
flags.DEFINE_string("results", None, "The path to the result.csv output from UbiPose")
flags.DEFINE_string("stats", None, "The path to the stats.csv output from UbiPose")

pd.set_option("display.max_rows", None)

BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array(
            [
                [Rxx - Ryy - Rzz, 0, 0, 0],
                [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
            ]
        )
        / 3.0
    )
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


class Image(BaseImage):
    def qvec2rotmat(self):
        return qvec2rotmat(self.qvec)


def read_images_text(path):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesText(const std::string& path)
        void Reconstruction::WriteImagesText(const std::string& path)
    """
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack(
                    [tuple(map(float, elems[0::3])), tuple(map(float, elems[1::3]))]
                )
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id,
                    qvec=qvec,
                    tvec=tvec,
                    camera_id=camera_id,
                    name=image_name,
                    xys=xys,
                    point3D_ids=point3D_ids,
                )
    return images


def computeErrorBetween(gt_q, gt_t, q, t):
    R_gt, t_gt = qvec2rotmat(gt_q), gt_t

    R, t = qvec2rotmat(q), t
    e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
    cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)
    e_R = np.rad2deg(np.abs(np.arccos(cos)))

    return e_R, e_t


def computeErrors(image, pred_qvec, pred_tvec):
    R = qvec2rotmat(pred_qvec)
    t = pred_tvec

    R_gt, t_gt = image.qvec2rotmat(), image.tvec
    e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
    cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1.0, 1.0)

    e_R = np.rad2deg(np.abs(np.arccos(cos)))

    return (e_R, e_t)


def parseResultFile(filename):
    predictions = {}
    with open(filename, "r") as f:
        for data in f.read().rstrip().split("\n"):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            predictions[name] = (qvec2rotmat(q), t)
    return predictions


def parseStatsFile(stats):
    stats_df = pd.read_csv(
        stats,
        sep=" ",
        header=None,
        index_col=False,
        names=[
            "filename",
            "vio_est_qvec_0",
            "vio_est_qvec_1",
            "vio_est_qvec_2",
            "vio_est_qvec_3",
            "vio_est_tvec_0",
            "vio_est_tvec_1",
            "vio_est_tvec_2",
            "num_projected_in_pose",
            "num_query_features",
            "num_total_mesh_features",
            "num_matches",
            "num_inliers",
            "preprocess_latency_ms",
            "superpoint_latency_ms",
            "superglue_latency_ms",
            "postprocess_latency_ms",
            "match_projection_latency_ms",
            "register_latency_ms",
            "total_latency_ms",
            "localized_qvec_0",
            "localized_qvec_1",
            "localized_qvec_2",
            "localized_qvec_3",
            "localized_tvec_0",
            "localized_tvec_1",
            "localized_tvec_2",
            "cache_localized",
            "cache_localized_num_query_features",
            "cache_localized_num_total_mesh_features",
            "cache_localized_num_projected_matched_features",
            "cache_localized_num_inliers",
            "cache_localized_preprocess_latency_ms",
            "cache_localized_superpoint_latency_ms",
            "cache_localized_superglue_latency_ms",
            "cache_localized_match_projection_latency_ms",
            "cache_localized_register_latency_ms",
            "cache_localized_total_latency_ms",
            "cache_localized_qvec_0",
            "cache_localized_qvec_1",
            "cache_localized_qvec_2",
            "cache_localized_qvec_3",
            "cache_localized_tvec_0",
            "cache_localized_tvec_1",
            "cache_localized_tvec_2",
            "early_exited",
            "localized",
            "accepted",
            "accepted_cache",
        ],
    )

    return stats_df


def main(argv):
    del argv

    images = read_images_text(FLAGS.colmap_image_txt)
    name2id = {image.name: i for i, image in images.items()}
    results = Path(FLAGS.results)
    stats = Path(FLAGS.stats)

    test_names = []
    for image_name in name2id.keys():
        # if image_name.startswith('left'):
        if not image_name.startswith("left") and not image_name.startswith("right"):
            test_names.append(image_name)
    list.sort(test_names)

    predictions = parseResultFile(results)

    q_list = []
    t_list = []

    errors_t = []
    errors_R = []
    nmi_list = []
    problem_list = []
    dfs = []

    for i in range(1, len(test_names)):
        name = test_names[i]
        image_filename = Path(name).name

        # Prediction
        if image_filename not in predictions.keys():
            continue
        R, t = predictions[image_filename]

        # Calculate error
        image = images[name2id[name]]
        e_R, e_t = computeErrors(image, rotmat2qvec(R), t)

        q_list.append(rotmat2qvec(R))
        t_list.append(t)

        errors_t.append(e_t)
        errors_R.append(e_R)

        df = pd.DataFrame([image_filename, e_R, e_t, image.qvec, image.tvec]).T
        dfs.append(df)

    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)

    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    out = f"Results for file {results.name}:"
    out += f"\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg"

    out += "\nPercentage of test images localized within:"
    threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f"\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%"

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(errors_R)
    ax[1].plot(errors_t)
    plt.savefig("localize_error.png")

    print(out)

    result_df = pd.concat(dfs, ignore_index=True)
    result_df.sort_values(by=1, inplace=True, ascending=False)
    result_df.rename(
        columns={0: "filename", 1: "error_R", 2: "error_t", 3: "gt_qvec", 4: "gt_tvec"},
        inplace=True,
    )

    fig, ax = plt.subplots(3, 2)
    fig.set_figwidth(20)
    fig.set_figheight(15)

    ser = result_df["error_t"]
    ser = ser.sort_values()

    cum_dist = np.linspace(0.0, 1.0, len(ser))
    ser_cdf = pd.Series(cum_dist, index=ser)
    ser_cdf.plot(drawstyle="steps", ax=ax[0][0])
    ax[0][0].set_title("CDF of translation error")
    ax[0][0].set_xlabel("Translation error (m)")
    ax[0][0].set_ylabel("CDF")

    ser = result_df["error_R"]
    ser = ser.sort_values()

    cum_dist = np.linspace(0.0, 1.0, len(ser))
    ser_cdf = pd.Series(cum_dist, index=ser)
    ser_cdf.plot(drawstyle="steps", ax=ax[0][1])
    ax[0][1].set_title("CDF of rotation error")
    ax[0][1].set_xlabel("Rotation error (deg)")
    ax[0][1].set_ylabel("CDF")

    ax[1][0].plot(errors_t)
    ax[1][0].set_title("Translation error")
    ax[1][0].set_ylabel("Translation error (m)")
    ax[1][0].set_xlabel("Frame")
    ax[1][0].set_xticks(np.arange(0, len(errors_t), 10))

    ax[1][1].plot(errors_R)
    ax[1][1].set_title("Rotation error")
    ax[1][1].set_ylabel("Rotation error (deg)")
    ax[1][1].set_xlabel("Frame")
    ax[1][1].set_xticks(np.arange(0, len(errors_R), 10))

    print("95th translation error = %f" % np.percentile(errors_t, 95))
    print("95th rotation error = %f" % np.percentile(errors_R, 95))

    print("99th translation error = %f" % np.percentile(errors_t, 99))
    print("99th rotation error = %f" % np.percentile(errors_R, 99))

    result_df.sort_values(by="filename", inplace=True)

    # print(result_df[result_df['error_R'] > 1])

    stats_df = parseStatsFile(stats)
    # stats_df.to_csv("stats_df.csv")

    merged_df = result_df.merge(stats_df, on="filename")
    merged_df["vio_error_q"] = 0
    merged_df["vio_error_t"] = 0
    merged_df["localized_error_q"] = 0
    merged_df["localized_error_t"] = 0
    merged_df["cache_localized_error_q"] = 0
    merged_df["cache_localized_error_t"] = 0
    merged_df["est_localized_error_q"] = 0
    merged_df["est_localized_error_t"] = 0
    merged_df["est_cache_localized_error_q"] = 0
    merged_df["est_cache_localized_error_t"] = 0

    for index, row in merged_df.iterrows():
        gt_qvec = row["gt_qvec"]
        gt_tvec = row["gt_tvec"]

        vio_qvec = [
            row["vio_est_qvec_0"],
            row["vio_est_qvec_1"],
            row["vio_est_qvec_2"],
            row["vio_est_qvec_3"],
        ]
        vio_tvec = [row["vio_est_tvec_0"], row["vio_est_tvec_1"], row["vio_est_tvec_2"]]

        vio_error_q, vio_error_t = computeErrorBetween(
            gt_qvec, gt_tvec, vio_qvec, vio_tvec
        )
        merged_df.at[index, "vio_error_q"] = vio_error_q
        merged_df.at[index, "vio_error_t"] = vio_error_t

        localized_qvec = [
            row["localized_qvec_0"],
            row["localized_qvec_1"],
            row["localized_qvec_2"],
            row["localized_qvec_3"],
        ]
        localized_tvec = [
            row["localized_tvec_0"],
            row["localized_tvec_1"],
            row["localized_tvec_2"],
        ]

        localized_error_q, localized_error_t = computeErrorBetween(
            gt_qvec, gt_tvec, localized_qvec, localized_tvec
        )
        merged_df.at[index, "localized_error_q"] = localized_error_q
        merged_df.at[index, "localized_error_t"] = localized_error_t

        cache_localized_qvec = [
            row["cache_localized_qvec_0"],
            row["cache_localized_qvec_1"],
            row["cache_localized_qvec_2"],
            row["cache_localized_qvec_3"],
        ]
        cache_localized_tvec = [
            row["cache_localized_tvec_0"],
            row["cache_localized_tvec_1"],
            row["cache_localized_tvec_2"],
        ]

        cache_localized_error_q, cache_localized_error_t = computeErrorBetween(
            gt_qvec, gt_tvec, cache_localized_qvec, cache_localized_tvec
        )
        merged_df.at[index, "cache_localized_error_q"] = cache_localized_error_q
        merged_df.at[index, "cache_localized_error_t"] = cache_localized_error_t

        est_localized_error_q, est_localized_error_t = computeErrorBetween(
            vio_qvec, vio_tvec, localized_qvec, localized_tvec
        )
        merged_df.at[index, "est_localized_error_q"] = est_localized_error_q
        merged_df.at[index, "est_localized_error_t"] = est_localized_error_t

        est_cache_localized_error_q, est_cache_localized_error_t = computeErrorBetween(
            vio_qvec, vio_tvec, cache_localized_qvec, cache_localized_tvec
        )
        merged_df.at[index, "est_cache_localized_error_q"] = est_cache_localized_error_q
        merged_df.at[index, "est_cache_localized_error_t"] = est_cache_localized_error_t

    merged_df["index"] = merged_df.index

    merged_df.plot(
        kind="scatter",
        x="index",
        y="est_localized_error_t",
        ax=ax[2][0],
        label="est localize error",
        color="y",
    )
    merged_df.plot(
        kind="scatter",
        x="index",
        y="est_cache_localized_error_t",
        ax=ax[2][0],
        label="est cache localize error",
        color="c",
    )
    merged_df.plot(
        kind="scatter",
        x="index",
        y="localized_error_t",
        ax=ax[2][0],
        label="localize error",
        color="r",
    )
    merged_df.plot(
        kind="scatter",
        x="index",
        y="cache_localized_error_t",
        ax=ax[2][0],
        label="cache localize error",
        color="g",
    )
    merged_df.plot(y="vio_error_t", ax=ax[2][0], color="k")
    merged_df.plot(y="error_t", ax=ax[2][0])
    ax[2][0].set_ylim(ax[1][0].get_ylim())

    merged_df.plot(
        kind="scatter",
        x="index",
        y="est_localized_error_q",
        ax=ax[2][1],
        label="est localize error",
        color="y",
    )
    merged_df.plot(
        kind="scatter",
        x="index",
        y="est_cache_localized_error_q",
        ax=ax[2][1],
        label="est cache localize error",
        color="c",
    )
    merged_df.plot(
        kind="scatter",
        x="index",
        y="localized_error_q",
        ax=ax[2][1],
        label="localize error",
        color="r",
    )
    merged_df.plot(
        kind="scatter",
        x="index",
        y="cache_localized_error_q",
        ax=ax[2][1],
        label="cache localize error",
        color="g",
    )
    merged_df.plot(y="vio_error_q", ax=ax[2][1], color="k")
    merged_df.plot(y="error_R", ax=ax[2][1])

    ax[2][1].set_ylim(ax[1][1].get_ylim())

    merged_df.to_csv("merged.csv")

    plt.savefig("localize_error.png")

    # print("Superpoint 95th latency %f ms" % stats_df.superpoint_ms.quantile(0.95))
    # print("Superglue 95th latency %f ms" % stats_df.superglue_ms.quantile(0.95))
    # print("Total 95th latency %f ms" % stats_df.total_ms.quantile(0.95))
    # print("Total median latency %f ms" % stats_df.total_ms.quantile(0.5))

    print(
        "Number of early exited frames %d" % len(stats_df[stats_df.early_exited == 1])
    )
    print(
        "Percentage of early exited frames %f"
        % (len(stats_df[stats_df.early_exited == 1]) / len(stats_df))
    )
    print(
        "Number of cache localized frames %d"
        % len(stats_df[stats_df.accepted_cache == 1])
    )
    print(
        "Percentage of cache localized frames %f"
        % (len(stats_df[stats_df.accepted_cache == 1]) / len(stats_df))
    )

    merged_df["pipeline_latency"] = np.where(
        merged_df["localized"] == 1,
        merged_df["total_latency_ms"],
        merged_df["cache_localized_total_latency_ms"],
    )
    print("Median latency = %.2fms, 95th latency = %.2fms" % (merged_df.pipeline_latency.quantile(0.5), merged_df.pipeline_latency.quantile(0.95)))

if __name__ == "__main__":
    app.run(main)
